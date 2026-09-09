"""API tests for XLSX knowledge-base upload validation and preview."""

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from openpyxl import Workbook

from backend.app.config.settings import Settings
from backend.app.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_WORKBOOK = PROJECT_ROOT / "knowledge" / "erp_faq.xlsx"
UPLOAD_URL = "/api/v1/knowledge/upload"


def workbook_bytes(
    rows: list[tuple[object, object]],
    headers: tuple[str, ...] = ("问题", "答案"),
) -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(headers)
    for row in rows:
        worksheet.append(row)
    output = BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()


def build_client(*, max_upload_mb: int = 10) -> TestClient:
    settings = Settings(
        app_env="testing",
        log_level="CRITICAL",
        knowledge_upload_max_mb=max_upload_mb,
        _env_file=None,
    )
    return TestClient(create_app(settings))


def test_upload_supplied_sample_returns_valid_preview() -> None:
    with SAMPLE_WORKBOOK.open("rb") as sample_file, build_client() as client:
        response = client.post(
            UPLOAD_URL,
            files={
                "file": (
                    "erp_faq.xlsx",
                    sample_file,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["file_name"] == "erp_faq.xlsx"
    assert payload["sheet_name"] == "Sheet1"
    assert payload["total_rows"] == 28
    assert payload["valid_rows"] == 28
    assert payload["invalid_rows"] == 0
    assert payload["is_valid"] is True
    assert payload["preview_count"] == 20
    assert payload["preview_truncated"] is True
    assert payload["records"][0]["question"] == "收银台/后台下载"
    assert payload["issues"] == []


def test_upload_valid_workbook_returns_importable_preview() -> None:
    content = workbook_bytes([("如何新增客户？", "进入客户管理后点击新增。")])

    with build_client() as client:
        response = client.post(
            UPLOAD_URL,
            files={"file": ("faq.xlsx", content, "application/octet-stream")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "file_name": "faq.xlsx",
        "sheet_name": "Sheet",
        "total_rows": 1,
        "valid_rows": 1,
        "invalid_rows": 0,
        "is_valid": True,
        "preview_count": 1,
        "preview_truncated": False,
        "records": [
            {
                "row_number": 2,
                "question": "如何新增客户？",
                "answer": "进入客户管理后点击新增。",
            }
        ],
        "issues": [],
    }


def test_upload_rejects_wrong_extension() -> None:
    with build_client() as client:
        response = client.post(
            UPLOAD_URL,
            files={"file": ("faq.xls", b"legacy-excel", "application/vnd.ms-excel")},
        )

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_file_type"


def test_upload_rejects_corrupt_xlsx() -> None:
    with build_client() as client:
        response = client.post(
            UPLOAD_URL,
            files={"file": ("faq.xlsx", b"not-an-xlsx", "application/octet-stream")},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_workbook"


def test_upload_rejects_invalid_headers() -> None:
    content = workbook_bytes([], headers=("question", "answer"))

    with build_client() as client:
        response = client.post(
            UPLOAD_URL,
            files={"file": ("faq.xlsx", content, "application/octet-stream")},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_headers"


def test_upload_rejects_empty_file() -> None:
    with build_client() as client:
        response = client.post(
            UPLOAD_URL,
            files={"file": ("faq.xlsx", b"", "application/octet-stream")},
        )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "empty_file"


def test_upload_rejects_file_above_configured_limit() -> None:
    oversized_content = b"x" * (1024 * 1024 + 1)

    with build_client(max_upload_mb=1) as client:
        response = client.post(
            UPLOAD_URL,
            files={"file": ("faq.xlsx", oversized_content, "application/octet-stream")},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == {
        "code": "file_too_large",
        "message": "上传文件不能超过 1 MB",
    }


def test_upload_requires_multipart_file_field() -> None:
    with build_client() as client:
        response = client.post(UPLOAD_URL)

    assert response.status_code == 422
