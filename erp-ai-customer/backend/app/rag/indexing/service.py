"""Orchestrate validated ERP FAQ records into a persistent vector store."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from backend.app.knowledge.excel_parser import parse_knowledge_workbook
from backend.app.knowledge.exceptions import KnowledgeWorkbookError
from backend.app.knowledge.models import KnowledgeParseResult, KnowledgeRecord
from backend.app.rag.embeddings.exceptions import EmbeddingError
from backend.app.rag.embeddings.service import EmbeddingService
from backend.app.rag.indexing.exceptions import (
    KnowledgeIndexEmbeddingError,
    KnowledgeIndexInputError,
    KnowledgeIndexResultError,
    KnowledgeIndexStorageError,
    KnowledgeIndexValidationError,
)
from backend.app.rag.indexing.models import KnowledgeIndexResult
from backend.app.rag.vector_store.base import VectorRecord, VectorStore
from backend.app.rag.vector_store.exceptions import VectorStoreError

DOCUMENT_TYPE = "erp_faq"
XLSX_SUFFIX = ".xlsx"


class KnowledgeIndexService:
    """Create deterministic FAQ documents, embeddings, and vector records."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    @property
    def embedding_service(self) -> EmbeddingService:
        """Return the validated embedding boundary used by the indexer."""
        return self._embedding_service

    @property
    def vector_store(self) -> VectorStore:
        """Return the provider-neutral vector store used by the indexer."""
        return self._vector_store

    def index_workbook(self, path: str | Path) -> KnowledgeIndexResult:
        """Parse and atomically index one valid XLSX knowledge workbook."""
        try:
            workbook_path = Path(path)
        except TypeError as exc:
            raise KnowledgeIndexInputError("workbook path must be a filesystem path") from exc

        try:
            parse_result = parse_knowledge_workbook(workbook_path)
        except KnowledgeWorkbookError as exc:
            raise KnowledgeIndexValidationError(
                f"Knowledge workbook cannot be indexed: {exc}"
            ) from exc

        return self.index_parse_result(parse_result, source_name=workbook_path.name)

    def index_parse_result(
        self,
        parse_result: KnowledgeParseResult,
        *,
        source_name: str | Path,
    ) -> KnowledgeIndexResult:
        """Index a previously parsed workbook without allowing partial imports."""
        if not isinstance(parse_result, KnowledgeParseResult):
            raise KnowledgeIndexInputError(
                "parse_result must be a KnowledgeParseResult instance"
            )

        normalized_source_name = _normalize_source_name(source_name)
        sheet_name = _normalize_required_text(
            parse_result.sheet_name,
            input_name="sheet name",
        )
        _validate_parse_summary(parse_result)
        if parse_result.issues:
            first_issue = parse_result.issues[0]
            location = (
                f"row {first_issue.row_number}"
                if first_issue.row_number is not None
                else "workbook"
            )
            raise KnowledgeIndexValidationError(
                f"Knowledge workbook has {len(parse_result.issues)} validation issue(s); "
                f"partial indexing is disabled. First issue at {location}: "
                f"{first_issue.message}"
            )

        prepared_records = _prepare_records(parse_result.records)
        documents = tuple(
            _build_document(question=question, answer=answer)
            for _, question, answer in prepared_records
        )

        try:
            embeddings = self.embedding_service.embed_documents(documents)
        except EmbeddingError as exc:
            raise KnowledgeIndexEmbeddingError(
                f"Failed to create embeddings for '{normalized_source_name}'"
            ) from exc
        except Exception as exc:
            raise KnowledgeIndexEmbeddingError(
                f"Unexpected embedding failure for '{normalized_source_name}'"
            ) from exc

        provider = self.embedding_service.provider
        record_ids = tuple(
            _build_record_id(
                source_name=normalized_source_name,
                sheet_name=sheet_name,
                question=question,
            )
            for _, question, _ in prepared_records
        )
        vector_records = tuple(
            VectorRecord(
                id=record_id,
                document=document,
                embedding=embedding,
                metadata={
                    "document_type": DOCUMENT_TYPE,
                    "source_file": normalized_source_name,
                    "sheet_name": sheet_name,
                    "row_number": record.row_number,
                    "question": question,
                    "embedding_provider": provider.provider_name,
                    "embedding_model": provider.model_name,
                    "embedding_dimensions": self.embedding_service.dimensions,
                },
            )
            for (record, question, _), record_id, document, embedding in zip(
                prepared_records,
                record_ids,
                documents,
                embeddings,
                strict=True,
            )
        )

        try:
            indexed_count = self.vector_store.upsert(vector_records)
        except VectorStoreError as exc:
            raise KnowledgeIndexStorageError(
                f"Failed to persist vectors in vector store collection "
                f"'{self.vector_store.collection_name}'"
            ) from exc
        except Exception as exc:
            raise KnowledgeIndexStorageError(
                f"Unexpected vector store failure for collection "
                f"'{self.vector_store.collection_name}'"
            ) from exc

        expected_count = len(vector_records)
        if (
            isinstance(indexed_count, bool)
            or not isinstance(indexed_count, int)
            or indexed_count != expected_count
        ):
            raise KnowledgeIndexResultError(
                f"Vector store reported {indexed_count!r} indexed records; "
                f"expected {expected_count}"
            )

        return KnowledgeIndexResult(
            source_name=normalized_source_name,
            sheet_name=sheet_name,
            total_rows=parse_result.total_rows,
            indexed_records=indexed_count,
            collection_name=self.vector_store.collection_name,
            embedding_provider=provider.provider_name,
            embedding_model=provider.model_name,
            embedding_dimensions=self.embedding_service.dimensions,
            record_ids=record_ids,
        )


def _normalize_source_name(source_name: str | Path) -> str:
    if isinstance(source_name, Path):
        raw_name = source_name.name
    elif isinstance(source_name, str):
        raw_name = Path(source_name.replace("\\", "/")).name
    else:
        raise KnowledgeIndexInputError("source name must be a string or path")

    normalized = raw_name.strip()
    if not normalized:
        raise KnowledgeIndexInputError("source name must not be blank")
    if Path(normalized).suffix.lower() != XLSX_SUFFIX:
        raise KnowledgeIndexInputError("source name must use the .xlsx extension")
    return normalized


def _validate_parse_summary(parse_result: KnowledgeParseResult) -> None:
    total_rows = parse_result.total_rows
    if isinstance(total_rows, bool) or not isinstance(total_rows, int) or total_rows < 0:
        raise KnowledgeIndexInputError("parse result total_rows must be a non-negative integer")
    if total_rows < len(parse_result.records):
        raise KnowledgeIndexInputError(
            "parse result total_rows cannot be smaller than its valid record count"
        )


def _prepare_records(
    records: tuple[KnowledgeRecord, ...],
) -> tuple[tuple[KnowledgeRecord, str, str], ...]:
    if not records:
        raise KnowledgeIndexInputError("knowledge workbook contains no valid records")

    prepared: list[tuple[KnowledgeRecord, str, str]] = []
    seen_questions: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, KnowledgeRecord):
            raise KnowledgeIndexInputError(
                f"records[{index}] must be a KnowledgeRecord"
            )
        if (
            isinstance(record.row_number, bool)
            or not isinstance(record.row_number, int)
            or record.row_number < 2
        ):
            raise KnowledgeIndexInputError(
                f"records[{index}].row_number must be an integer greater than or equal to 2"
            )

        question = _normalize_required_text(
            record.question,
            input_name=f"records[{index}].question",
        )
        answer = _normalize_required_text(
            record.answer,
            input_name=f"records[{index}].answer",
        )
        question_key = " ".join(question.split()).casefold()
        if question_key in seen_questions:
            raise KnowledgeIndexInputError(
                f"records[{index}].question duplicates another knowledge record"
            )
        seen_questions.add(question_key)
        prepared.append((record, question, answer))
    return tuple(prepared)


def _normalize_required_text(value: str, *, input_name: str) -> str:
    if not isinstance(value, str):
        raise KnowledgeIndexInputError(f"{input_name} must be a string")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise KnowledgeIndexInputError(f"{input_name} must not be blank")
    return normalized


def _build_document(*, question: str, answer: str) -> str:
    return f"问题：{question}\n答案：{answer}"


def _build_record_id(*, source_name: str, sheet_name: str, question: str) -> str:
    canonical_identity = "\x1f".join(
        (
            source_name.casefold(),
            sheet_name.casefold(),
            " ".join(question.split()).casefold(),
        )
    )
    digest = sha256(canonical_identity.encode("utf-8")).hexdigest()
    return f"faq-{digest}"
