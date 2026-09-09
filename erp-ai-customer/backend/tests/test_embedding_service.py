"""Unit tests for the provider-neutral embedding contract and service."""

from collections.abc import Sequence

import pytest

from backend.app.rag.embeddings.base import EmbeddingProvider
from backend.app.rag.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingInputError,
    EmbeddingResultError,
)
from backend.app.rag.embeddings.service import EmbeddingService


class RecordingProvider:
    """Small deterministic provider used only to verify the service contract."""

    provider_name = "test"
    model_name = "test-model"
    dimensions = 3

    def __init__(self) -> None:
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []
        self.document_result: Sequence[Sequence[float]] | None = None
        self.query_result: Sequence[float] = (9.0, 8.0, 7.0)

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        normalized_texts = tuple(texts)
        self.document_calls.append(normalized_texts)
        if self.document_result is not None:
            return self.document_result
        return [
            (float(index), float(len(text)), 1.0)
            for index, text in enumerate(normalized_texts, start=1)
        ]

    def embed_query(self, text: str) -> Sequence[float]:
        self.query_calls.append(text)
        return self.query_result


def test_provider_implements_runtime_contract() -> None:
    assert isinstance(RecordingProvider(), EmbeddingProvider)


def test_embed_documents_normalizes_inputs_and_returns_immutable_vectors() -> None:
    provider = RecordingProvider()
    service = EmbeddingService(provider)

    vectors = service.embed_documents(["  ERP 问题一  ", "ERP 问题二\n"])

    assert provider.document_calls == [("ERP 问题一", "ERP 问题二")]
    assert vectors == ((1.0, 7.0, 1.0), (2.0, 7.0, 1.0))
    assert isinstance(vectors, tuple)
    assert all(isinstance(vector, tuple) for vector in vectors)


def test_document_batch_rejects_a_single_string_value() -> None:
    provider = RecordingProvider()
    service = EmbeddingService(provider)

    with pytest.raises(EmbeddingInputError, match="sequence of strings"):
        service.embed_documents("一个完整问题")

    assert provider.document_calls == []


def test_empty_document_batch_returns_without_calling_provider() -> None:
    provider = RecordingProvider()
    service = EmbeddingService(provider)

    assert service.embed_documents([]) == ()
    assert provider.document_calls == []


def test_embed_query_uses_query_specific_provider_method() -> None:
    provider = RecordingProvider()
    service = EmbeddingService(provider)

    vector = service.embed_query("  如何新增商品？  ")

    assert vector == (9.0, 8.0, 7.0)
    assert provider.query_calls == ["如何新增商品？"]
    assert provider.document_calls == []


@pytest.mark.parametrize("invalid_text", ["", "  \t\n  ", 123, None])
def test_blank_or_non_string_input_is_rejected(invalid_text: object) -> None:
    provider = RecordingProvider()
    service = EmbeddingService(provider)

    with pytest.raises(EmbeddingInputError):
        service.embed_documents([invalid_text])  # type: ignore[list-item]

    assert provider.document_calls == []


def test_provider_result_count_must_match_input_count() -> None:
    provider = RecordingProvider()
    provider.document_result = [(1.0, 2.0, 3.0)]
    service = EmbeddingService(provider)

    with pytest.raises(EmbeddingResultError, match="1 vectors for 2 input texts"):
        service.embed_documents(["问题一", "问题二"])


def test_provider_vector_dimensions_are_enforced() -> None:
    provider = RecordingProvider()
    provider.document_result = [(1.0, 2.0)]
    service = EmbeddingService(provider)

    with pytest.raises(EmbeddingResultError, match="2 dimensions; expected 3"):
        service.embed_documents(["问题"])


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), True, "not-number"])
def test_provider_vector_values_must_be_finite_numbers(invalid_value: object) -> None:
    provider = RecordingProvider()
    provider.query_result = (1.0, invalid_value, 3.0)  # type: ignore[assignment]
    service = EmbeddingService(provider)

    with pytest.raises(EmbeddingResultError):
        service.embed_query("测试查询")


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("provider_name", "  "),
        ("model_name", ""),
        ("dimensions", 0),
        ("dimensions", True),
    ],
)
def test_invalid_provider_metadata_fails_at_service_creation(
    attribute: str,
    value: object,
) -> None:
    provider = RecordingProvider()
    setattr(provider, attribute, value)

    with pytest.raises(EmbeddingConfigurationError):
        EmbeddingService(provider)
