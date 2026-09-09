"""Unit tests for deterministic ERP FAQ vector indexing."""

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

import pytest

from backend.app.knowledge.excel_parser import parse_knowledge_workbook
from backend.app.knowledge.models import (
    KnowledgeParseResult,
    KnowledgeRecord,
    KnowledgeValidationIssue,
)
from backend.app.rag.embeddings.base import EmbeddingProvider
from backend.app.rag.embeddings.exceptions import EmbeddingInferenceError
from backend.app.rag.embeddings.service import EmbeddingService
from backend.app.rag.indexing.exceptions import (
    KnowledgeIndexEmbeddingError,
    KnowledgeIndexInputError,
    KnowledgeIndexResultError,
    KnowledgeIndexStorageError,
)
from backend.app.rag.indexing.service import KnowledgeIndexService
from backend.app.rag.vector_store.base import VectorRecord, VectorSearchResult, VectorStore
from backend.app.rag.vector_store.exceptions import VectorStoreOperationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_WORKBOOK = PROJECT_ROOT / "knowledge" / "erp_faq.xlsx"


class RecordingEmbeddingProvider:
    """Deterministic provider used to verify indexing orchestration."""

    provider_name = "test-embedding"
    model_name = "test-model"
    dimensions = 3

    def __init__(self) -> None:
        self.document_calls: list[tuple[str, ...]] = []
        self.error: Exception | None = None

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        normalized_texts = tuple(texts)
        self.document_calls.append(normalized_texts)
        if self.error is not None:
            raise self.error
        return [
            (float(index), float(len(text)), 1.0)
            for index, text in enumerate(normalized_texts, start=1)
        ]

    def embed_query(self, text: str) -> Sequence[float]:
        raise AssertionError(f"Query embedding is not used while indexing: {text}")


class RecordingVectorStore:
    """Provider-neutral vector-store replacement that records upserts."""

    provider_name = "test-vector-store"
    collection_name = "erp_faq_test"

    def __init__(self) -> None:
        self.upsert_calls: list[tuple[VectorRecord, ...]] = []
        self.upsert_result: object | None = None
        self.error: Exception | None = None

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        normalized_records = tuple(records)
        self.upsert_calls.append(normalized_records)
        if self.error is not None:
            raise self.error
        if self.upsert_result is not None:
            return self.upsert_result  # type: ignore[return-value]
        return len(normalized_records)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int | None = None,
        where: Mapping[str, object] | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        raise AssertionError(
            f"Query is not used while indexing: {query_embedding}, {limit}, {where}"
        )

    def delete(self, ids: Sequence[str]) -> None:
        raise AssertionError(f"Delete is not used while indexing: {ids}")

    def count(self) -> int:
        raise AssertionError("Count is not used while indexing")


def make_parse_result(
    records: tuple[KnowledgeRecord, ...] | None = None,
    *,
    issues: tuple[KnowledgeValidationIssue, ...] = (),
    sheet_name: str = "Sheet1",
) -> KnowledgeParseResult:
    active_records = records if records is not None else (
        KnowledgeRecord(
            row_number=2,
            question="如何新增客户？",
            answer="进入客户管理后点击新增。",
        ),
        KnowledgeRecord(
            row_number=3,
            question="如何查询库存？",
            answer="进入库存查询页面选择仓库。",
        ),
    )
    return KnowledgeParseResult(
        sheet_name=sheet_name,
        total_rows=len(active_records) + len(issues),
        records=active_records,
        issues=issues,
    )


def build_service() -> tuple[
    KnowledgeIndexService,
    RecordingEmbeddingProvider,
    RecordingVectorStore,
]:
    provider = RecordingEmbeddingProvider()
    vector_store = RecordingVectorStore()
    service = KnowledgeIndexService(
        embedding_service=EmbeddingService(provider),
        vector_store=vector_store,
    )
    return service, provider, vector_store


def expected_record_id(source_name: str, sheet_name: str, question: str) -> str:
    canonical_identity = "".join(
        (
            source_name.casefold(),
            sheet_name.casefold(),
            " ".join(question.split()).casefold(),
        )
    )
    return f"faq-{sha256(canonical_identity.encode('utf-8')).hexdigest()}"


def test_test_doubles_implement_provider_neutral_contracts() -> None:
    provider = RecordingEmbeddingProvider()
    vector_store = RecordingVectorStore()

    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(vector_store, VectorStore)


def test_index_parse_result_embeds_documents_and_upserts_vector_records() -> None:
    service, provider, vector_store = build_service()
    parse_result = make_parse_result()

    result = service.index_parse_result(
        parse_result,
        source_name=r"C:\uploads\ERP_FAQ.XLSX",
    )

    documents = (
        "问题：如何新增客户？\n答案：进入客户管理后点击新增。",
        "问题：如何查询库存？\n答案：进入库存查询页面选择仓库。",
    )
    assert provider.document_calls == [documents]
    assert len(vector_store.upsert_calls) == 1

    indexed_records = vector_store.upsert_calls[0]
    expected_ids = tuple(
        expected_record_id("ERP_FAQ.XLSX", "Sheet1", record.question)
        for record in parse_result.records
    )
    assert tuple(record.id for record in indexed_records) == expected_ids
    assert tuple(record.document for record in indexed_records) == documents
    assert tuple(record.embedding for record in indexed_records) == (
        (1.0, float(len(documents[0])), 1.0),
        (2.0, float(len(documents[1])), 1.0),
    )
    assert dict(indexed_records[0].metadata) == {
        "document_type": "erp_faq",
        "source_file": "ERP_FAQ.XLSX",
        "sheet_name": "Sheet1",
        "row_number": 2,
        "question": "如何新增客户？",
        "embedding_provider": "test-embedding",
        "embedding_model": "test-model",
        "embedding_dimensions": 3,
    }
    assert result.source_name == "ERP_FAQ.XLSX"
    assert result.sheet_name == "Sheet1"
    assert result.total_rows == 2
    assert result.indexed_records == 2
    assert result.collection_name == "erp_faq_test"
    assert result.embedding_provider == "test-embedding"
    assert result.embedding_model == "test-model"
    assert result.embedding_dimensions == 3
    assert result.record_ids == expected_ids


def test_record_ids_are_stable_when_answers_or_row_positions_change() -> None:
    service, _, vector_store = build_service()
    first = make_parse_result(
        records=(KnowledgeRecord(2, "如何新增客户？", "旧答案"),)
    )
    second = make_parse_result(
        records=(KnowledgeRecord(20, " 如何新增客户？ ", "新答案"),)
    )

    first_result = service.index_parse_result(first, source_name="erp_faq.xlsx")
    second_result = service.index_parse_result(second, source_name="ERP_FAQ.XLSX")

    assert first_result.record_ids == second_result.record_ids
    assert vector_store.upsert_calls[0][0].document.endswith("旧答案")
    assert vector_store.upsert_calls[1][0].document.endswith("新答案")


def test_different_sources_create_different_record_ids() -> None:
    service, _, _ = build_service()
    parse_result = make_parse_result(records=(KnowledgeRecord(2, "相同问题", "答案"),))

    first = service.index_parse_result(parse_result, source_name="erp_faq.xlsx")
    second = service.index_parse_result(parse_result, source_name="other_faq.xlsx")

    assert first.record_ids != second.record_ids


def test_supplied_sample_is_indexed_as_one_atomic_import() -> None:
    service, provider, vector_store = build_service()
    parse_result = parse_knowledge_workbook(SAMPLE_WORKBOOK)

    result = service.index_parse_result(
        parse_result,
        source_name=SAMPLE_WORKBOOK.name,
    )

    assert result.total_rows == 28
    assert result.indexed_records == 28
    assert len(result.record_ids) == 28
    assert len(provider.document_calls) == 1
    assert len(provider.document_calls[0]) == 28
    assert len(vector_store.upsert_calls) == 1
    assert len(vector_store.upsert_calls[0]) == 28


def test_index_workbook_parses_and_indexes_the_supplied_sample() -> None:
    service, provider, vector_store = build_service()

    result = service.index_workbook(SAMPLE_WORKBOOK)

    assert result.total_rows == 28
    assert result.indexed_records == 28
    assert len(provider.document_calls) == 1
    assert len(vector_store.upsert_calls) == 1


@pytest.mark.parametrize(
    ("parse_result", "source_name"),
    [
        (make_parse_result(records=()), "erp_faq.xlsx"),
        (make_parse_result(sheet_name="  "), "erp_faq.xlsx"),
        (make_parse_result(), "  "),
        (make_parse_result(), "faq.csv"),
        (
            make_parse_result(records=(KnowledgeRecord(1, "问题", "答案"),)),
            "erp_faq.xlsx",
        ),
        (
            make_parse_result(records=(KnowledgeRecord(2, "  ", "答案"),)),
            "erp_faq.xlsx",
        ),
        (
            make_parse_result(records=(KnowledgeRecord(2, "问题", "  "),)),
            "erp_faq.xlsx",
        ),
        (
            make_parse_result(
                records=(
                    KnowledgeRecord(2, "重复问题", "答案一"),
                    KnowledgeRecord(3, " 重复问题 ", "答案二"),
                )
            ),
            "erp_faq.xlsx",
        ),
    ],
)
def test_invalid_index_inputs_fail_before_embedding_or_storage(
    parse_result: KnowledgeParseResult,
    source_name: str,
) -> None:
    service, provider, vector_store = build_service()

    with pytest.raises(KnowledgeIndexInputError):
        service.index_parse_result(parse_result, source_name=source_name)

    assert provider.document_calls == []
    assert vector_store.upsert_calls == []


def test_embedding_failures_are_wrapped_without_calling_vector_store() -> None:
    service, provider, vector_store = build_service()
    provider.error = EmbeddingInferenceError("model inference failed")

    with pytest.raises(KnowledgeIndexEmbeddingError, match="embedding") as error:
        service.index_parse_result(make_parse_result(), source_name="erp_faq.xlsx")

    assert error.value.__cause__ is provider.error
    assert vector_store.upsert_calls == []


def test_vector_store_failures_are_wrapped() -> None:
    service, _, vector_store = build_service()
    vector_store.error = VectorStoreOperationError("database unavailable")

    with pytest.raises(KnowledgeIndexStorageError, match="vector store") as error:
        service.index_parse_result(make_parse_result(), source_name="erp_faq.xlsx")

    assert error.value.__cause__ is vector_store.error


@pytest.mark.parametrize("upsert_result", [0, True, "2"])
def test_invalid_vector_store_counts_are_rejected(upsert_result: object) -> None:
    service, _, vector_store = build_service()
    vector_store.upsert_result = upsert_result

    with pytest.raises(KnowledgeIndexResultError, match="reported"):
        service.index_parse_result(make_parse_result(), source_name="erp_faq.xlsx")
