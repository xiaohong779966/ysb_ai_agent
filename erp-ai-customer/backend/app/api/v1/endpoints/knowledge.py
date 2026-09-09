"""Upload and validate an ERP FAQ workbook without persisting or indexing it."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from backend.app.config.settings import Settings
from backend.app.knowledge.excel_parser import parse_knowledge_bytes
from backend.app.knowledge.exceptions import (
    InvalidKnowledgeHeadersError,
    InvalidWorkbookError,
    UnsupportedWorkbookTypeError,
)
from backend.app.schemas.knowledge import (
    KnowledgeRecordResponse,
    KnowledgeUploadResponse,
    KnowledgeValidationIssueResponse,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
PREVIEW_RECORD_LIMIT = 20


@router.post(
    "/upload",
    response_model=KnowledgeUploadResponse,
    summary="Validate and preview an ERP FAQ workbook",
    status_code=status.HTTP_200_OK,
)
async def upload_knowledge_workbook(
    request: Request,
    file: Annotated[
        UploadFile,
        File(description="ERP FAQ workbook in .xlsx format"),
    ],
) -> KnowledgeUploadResponse:
    """Validate an uploaded XLSX file and return valid rows plus validation issues."""
    settings: Settings = request.app.state.settings
    filename = _safe_filename(file.filename)

    try:
        content = await file.read(settings.knowledge_upload_max_bytes + 1)
    finally:
        await file.close()

    if not content:
        raise _http_error(
            status.HTTP_400_BAD_REQUEST,
            "empty_file",
            "上传文件不能为空",
        )
    if len(content) > settings.knowledge_upload_max_bytes:
        raise _http_error(
            status.HTTP_413_CONTENT_TOO_LARGE,
            "file_too_large",
            f"上传文件不能超过 {settings.knowledge_upload_max_mb} MB",
        )

    try:
        result = await run_in_threadpool(
            parse_knowledge_bytes,
            content,
            filename=filename,
        )
    except UnsupportedWorkbookTypeError as exc:
        raise _http_error(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "unsupported_file_type",
            str(exc),
        ) from exc
    except InvalidKnowledgeHeadersError as exc:
        raise _http_error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "invalid_headers",
            str(exc),
        ) from exc
    except InvalidWorkbookError as exc:
        raise _http_error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "invalid_workbook",
            str(exc),
        ) from exc

    preview_records = result.records[:PREVIEW_RECORD_LIMIT]
    return KnowledgeUploadResponse(
        file_name=filename,
        sheet_name=result.sheet_name,
        total_rows=result.total_rows,
        valid_rows=result.valid_rows,
        invalid_rows=result.invalid_rows,
        is_valid=result.is_valid,
        preview_count=len(preview_records),
        preview_truncated=result.valid_rows > len(preview_records),
        records=[
            KnowledgeRecordResponse(
                row_number=record.row_number,
                question=record.question,
                answer=record.answer,
            )
            for record in preview_records
        ],
        issues=[
            KnowledgeValidationIssueResponse(
                row_number=issue.row_number,
                field=issue.field,
                code=issue.code,
                message=issue.message,
            )
            for issue in result.issues
        ],
    )


def _safe_filename(filename: str | None) -> str:
    if not filename or not filename.strip():
        raise _http_error(
            status.HTTP_400_BAD_REQUEST,
            "missing_filename",
            "上传文件必须包含文件名",
        )
    return Path(filename.replace("\\", "/")).name


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )

