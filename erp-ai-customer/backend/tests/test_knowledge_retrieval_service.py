"""Unit tests for ERP FAQ similarity retrieval orchestration."""

from collections.abc import Mapping, Sequence
from math import nan

import pytest

from backend.app.rag.embeddings.base import EmbeddingProvider
from backend.app.rag.embeddings.exceptions import EmbeddingInferenceError
from backend.app.rag.embeddings.service import EmbeddingService
from backend.app.rag.retrieval.exceptions import (
    KnowledgeRetrievalConfigurationError,
    KnowledgeRetrievalEmbeddingError,
    KnowledgeRetrievalInputError,
    KnowledgeRetrievalResultError,
    KnowledgeRetrievalStorageError,
)
from backend.app.rag.retrieval.service import KnowledgeRetrievalService
from backend.app.rag.vector_store.base import VectorRecord, VectorSearchResult, VectorStore
from backend.app.rag.vector_store.exceptions import VectorStoreOperationError

EXPECTED_FILTER = {
    "$and": [
        {"document_type": "erp_faq"},
        {"embedding_provider": "test-embedding"},
        {"embedding_model": "test-model"},
        {"embedding_dimensions": 3},
    ]
}


class RecordingEmbeddingProvider:
    """Deterministic query provider that records retrieval calls."""

    provider_name = "test-embedding"
    model_name = "test-model"
    dimensions = 3

    def __init__(self) -> None:
        self.query_calls: list[str] = []
        self.error: Exception | None = None

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise AssertionError(f"Document embedding is not used for retrieval: {texts}")

    def embed_query(self, text: str) -> Sequence[float]:
        self.query_calls.append(text)
        if self.error is not None:
            raise self.error
        return (0.1, 0.2, 0.3)


class RecordingVectorStore:
    """Provider-neutral vector-store replacement used by retrieval tests."""

    provider_name = "test-vector-store"
    collection_name = "erp_faq_test"

    def __init__(self) -> None:
        self.query_calls: list[dict[str, object]] = []
        self.query_result: object = ()
        self.error: Exception | None = None

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        raise AssertionError(f"Upsert is not used for retrieval: {records}")

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int | None = None,
        where: Mapping[str, object] | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        self.query_calls.append(
            {
                "query_embedding": tuple(query_embedding),
                "limit": limit,
                "where": where,
            }
        )
        if self.error is not None:
            raise self.error
        return self.query_result  # type: ignore[return-value]

    def delete(self, ids: Sequence[str]) -> None:
        raise AssertionError(f"Delete is not used for retrieval: {ids}")

    def count(self) -> int:
        raise AssertionError("Count is not used for retrieval")


def make_metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "document_type": "erp_faq",
        "source_file": "erp_faq.xlsx",
        "sheet_name": "ERP FAQ",
        "row_number": 2,
        "question": "如何新增客户？",
        "embedding_provider": "test-embedding",
        "embedding_model": "test-model",
        "embedding_dimensions": 3,
    }
    metadata.update(overrides)
    return metadata


def make_hit(
    *,
    record_id: object = "faq-1",
    document: object = "问题：如何新增客户？\n答案：进入客户管理后点击新增。",
    metadata: object | None = None,
    distance: object = 0.1,
) -> VectorSearchResult:
    active_metadata = make_metadata() if metadata is None else metadata
    return VectorSearchResult(
        id=record_id,  # type: ignore[arg-type]
        document=document,  # type: ignore[arg-type]
        metadata=active_metadata,  # type: ignore[arg-type]
        distance=distance,  # type: ignore[arg-type]
    )


def build_service(
    *,
    default_limit: int = 5,
) -> tuple[
    KnowledgeRetrievalService,
    RecordingEmbeddingProvider,
    RecordingVectorStore,
]:
    provider = RecordingEmbeddingProvider()
    vector_store = RecordingVectorStore()
    service = KnowledgeRetrievalService(
        embedding_service=EmbeddingService(provider),
        vector_store=vector_store,
        default_limit=default_limit,
    )
    return service, provider, vector_store


def test_test_doubles_implement_provider_neutral_contracts() -> None:
    provider = RecordingEmbeddingProvider()
    vector_store = RecordingVectorStore()

    assert isinstance(provider, EmbeddingProvider)
    assert isinstance(vector_store, VectorStore)


def test_search_embeds_normalized_query_and_returns_domain_hit() -> None:
    service, provider, vector_store = build_service(default_limit=7)
    vector_store.query_result = (make_hit(),)

    result = service.search("  如何新增客户？\r\n  ")

    assert provider.query_calls == ["如何新增客户？"]
    assert vector_store.query_calls == [
        {
            "query_embedding": (0.1, 0.2, 0.3),
            "limit": 7,
            "where": EXPECTED_FILTER,
        }
    ]
    assert result.query == "如何新增客户？"
    assert result.limit == 7
    assert result.collection_name == "erp_faq_test"
    assert result.embedding_provider == "test-embedding"
    assert result.embedding_model == "test-model"
    assert result.embedding_dimensions == 3
    assert result.total_hits == 1
    hit = result.hits[0]
    assert hit.record_id == "faq-1"
    assert hit.question == "如何新增客户？"
    assert hit.answer == "进入客户管理后点击新增。"
    assert hit.source_file == "erp_faq.xlsx"
    assert hit.sheet_name == "ERP FAQ"
    assert hit.row_number == 2
    assert hit.distance == 0.1


def test_search_limit_override_is_forwarded() -> None:
    service, _, vector_store = build_service(default_limit=5)

    result = service.search("库存如何查询？", limit=2)

    assert result.limit == 2
    assert vector_store.query_calls[0]["limit"] == 2


def test_empty_vector_store_result_is_a_successful_search() -> None:
    service, _, _ = build_service()

    result = service.search("没有匹配的问题")

    assert result.hits == ()
    assert result.total_hits == 0


@pytest.mark.parametrize("query", [None, 123, "", " \r\n "])
def test_invalid_queries_fail_before_embedding_or_storage(query: object) -> None:
    service, provider, vector_store = build_service()

    with pytest.raises(KnowledgeRetrievalInputError):
        service.search(query)  # type: ignore[arg-type]

    assert provider.query_calls == []
    assert vector_store.query_calls == []


@pytest.mark.parametrize("default_limit", [0, 101, True, 1.5, "5"])
def test_invalid_default_limits_are_rejected(default_limit: object) -> None:
    provider = RecordingEmbeddingProvider()
    vector_store = RecordingVectorStore()

    with pytest.raises(KnowledgeRetrievalConfigurationError):
        KnowledgeRetrievalService(
            embedding_service=EmbeddingService(provider),
            vector_store=vector_store,
            default_limit=default_limit,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("limit", [0, 101, True, 1.5, "5"])
def test_invalid_request_limits_fail_before_embedding_or_storage(limit: object) -> None:
    service, provider, vector_store = build_service()

    with pytest.raises(KnowledgeRetrievalInputError):
        service.search("如何新增客户？", limit=limit)  # type: ignore[arg-type]

    assert provider.query_calls == []
    assert vector_store.query_calls == []


def test_embedding_failures_are_wrapped_without_querying_vector_store() -> None:
    service, provider, vector_store = build_service()
    provider.error = EmbeddingInferenceError("model inference failed")

    with pytest.raises(KnowledgeRetrievalEmbeddingError, match="embedding") as error:
        service.search("如何新增客户？")

    assert error.value.__cause__ is provider.error
    assert vector_store.query_calls == []


def test_unexpected_embedding_failures_are_wrapped() -> None:
    service, provider, vector_store = build_service()
    provider.error = RuntimeError("unexpected model failure")

    with pytest.raises(KnowledgeRetrievalEmbeddingError, match="Unexpected") as error:
        service.search("如何新增客户？")

    assert error.value.__cause__ is provider.error
    assert vector_store.query_calls == []


def test_vector_store_failures_are_wrapped() -> None:
    service, _, vector_store = build_service()
    vector_store.error = VectorStoreOperationError("database unavailable")

    with pytest.raises(KnowledgeRetrievalStorageError, match="vector store") as error:
        service.search("如何新增客户？")

    assert error.value.__cause__ is vector_store.error


def test_unexpected_vector_store_failures_are_wrapped() -> None:
    service, _, vector_store = build_service()
    vector_store.error = RuntimeError("unexpected database failure")

    with pytest.raises(KnowledgeRetrievalStorageError, match="Unexpected") as error:
        service.search("如何新增客户？")

    assert error.value.__cause__ is vector_store.error


@pytest.mark.parametrize("query_result", [None, 1, "invalid"])
def test_non_iterable_or_text_results_are_rejected(query_result: object) -> None:
    service, _, vector_store = build_service()
    vector_store.query_result = query_result

    with pytest.raises(KnowledgeRetrievalResultError):
        service.search("如何新增客户？")


def test_result_count_cannot_exceed_requested_limit() -> None:
    service, _, vector_store = build_service()
    vector_store.query_result = (make_hit(record_id="faq-1"), make_hit(record_id="faq-2"))

    with pytest.raises(KnowledgeRetrievalResultError, match="limit of 1"):
        service.search("如何新增客户？", limit=1)


def test_duplicate_record_ids_are_rejected() -> None:
    service, _, vector_store = build_service()
    vector_store.query_result = (make_hit(), make_hit())

    with pytest.raises(KnowledgeRetrievalResultError, match="duplicate"):
        service.search("如何新增客户？")


@pytest.mark.parametrize(
    "query_result",
    [
        (object(),),
        (make_hit(record_id=" "),),
        (make_hit(document=" "),),
        (make_hit(metadata="invalid"),),
        (make_hit(metadata=make_metadata(document_type="other")),),
        (make_hit(metadata=make_metadata(source_file="faq.csv")),),
        (make_hit(metadata=make_metadata(sheet_name=" ")),),
        (make_hit(metadata=make_metadata(question=" ")),),
        (make_hit(metadata=make_metadata(row_number=1)),),
        (make_hit(metadata=make_metadata(embedding_provider="other")),),
        (make_hit(metadata=make_metadata(embedding_model="other")),),
        (make_hit(metadata=make_metadata(embedding_dimensions=4)),),
        (make_hit(document="问题：其他问题\n答案：答案"),),
        (make_hit(document="问题：如何新增客户？\n答案：  "),),
        (make_hit(distance=True),),
        (make_hit(distance=nan),),
    ],
)
def test_malformed_or_incompatible_hits_are_rejected(query_result: object) -> None:
    service, _, vector_store = build_service()
    vector_store.query_result = query_result

    with pytest.raises(KnowledgeRetrievalResultError):
        service.search("如何新增客户？")


def test_search_preserves_vector_store_ranking_order() -> None:
    service, _, vector_store = build_service()
    first = make_hit(record_id="faq-1", distance=0.05)
    second = make_hit(
        record_id="faq-2",
        document="问题：如何查询库存？\n答案：进入库存查询页面。",
        metadata=make_metadata(
            row_number=3,
            question="如何查询库存？",
        ),
        distance=0.2,
    )
    vector_store.query_result = (first, second)

    result = service.search("客户查询", limit=2)

    assert [hit.record_id for hit in result.hits] == ["faq-1", "faq-2"]
    assert [hit.distance for hit in result.hits] == [0.05, 0.2]
