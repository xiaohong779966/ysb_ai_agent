"""Validation boundary around concrete text embedding providers."""

from collections.abc import Sequence
from math import isfinite

from backend.app.rag.embeddings.base import (
    EmbeddingBatch,
    EmbeddingProvider,
    EmbeddingVector,
)
from backend.app.rag.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingInputError,
    EmbeddingResultError,
)


class EmbeddingService:
    """Normalize text inputs and enforce one stable vector contract."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._validate_provider_metadata()

    @property
    def provider(self) -> EmbeddingProvider:
        """Return the configured provider without exposing mutable service state."""
        return self._provider

    @property
    def dimensions(self) -> int:
        """Return the vector dimension guaranteed by this service."""
        return self._provider.dimensions

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed a document batch and validate count, dimensions, and values."""
        if isinstance(texts, (str, bytes)):
            raise EmbeddingInputError("texts must be a sequence of strings")
        normalized_texts = tuple(
            _normalize_text(text, input_name=f"texts[{index}]")
            for index, text in enumerate(texts)
        )
        if not normalized_texts:
            return ()

        raw_vectors = self._provider.embed_documents(normalized_texts)
        return _validate_batch(
            raw_vectors,
            expected_count=len(normalized_texts),
            expected_dimensions=self.dimensions,
        )

    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed one search query and validate its vector shape and values."""
        normalized_text = _normalize_text(text, input_name="query")
        raw_vector = self._provider.embed_query(normalized_text)
        return _validate_vector(
            raw_vector,
            expected_dimensions=self.dimensions,
            vector_name="query vector",
        )

    def _validate_provider_metadata(self) -> None:
        if (
            not isinstance(self._provider.provider_name, str)
            or not self._provider.provider_name.strip()
        ):
            raise EmbeddingConfigurationError("Embedding provider name must not be blank")
        if not isinstance(self._provider.model_name, str) or not self._provider.model_name.strip():
            raise EmbeddingConfigurationError("Embedding model name must not be blank")
        if isinstance(self._provider.dimensions, bool) or not isinstance(
            self._provider.dimensions, int
        ):
            raise EmbeddingConfigurationError("Embedding dimensions must be an integer")
        if self._provider.dimensions <= 0:
            raise EmbeddingConfigurationError("Embedding dimensions must be greater than zero")


def _normalize_text(text: str, *, input_name: str) -> str:
    if not isinstance(text, str):
        raise EmbeddingInputError(f"{input_name} must be a string")
    normalized = text.strip()
    if not normalized:
        raise EmbeddingInputError(f"{input_name} must not be blank")
    return normalized


def _validate_batch(
    raw_vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
    expected_dimensions: int,
) -> EmbeddingBatch:
    if isinstance(raw_vectors, (str, bytes)):
        raise EmbeddingResultError("Embedding provider returned an invalid vector batch")
    try:
        vectors = tuple(raw_vectors)
    except TypeError as exc:
        raise EmbeddingResultError(
            "Embedding provider returned a non-iterable vector batch"
        ) from exc

    if len(vectors) != expected_count:
        raise EmbeddingResultError(
            "Embedding provider returned "
            f"{len(vectors)} vectors for {expected_count} input texts"
        )
    return tuple(
        _validate_vector(
            vector,
            expected_dimensions=expected_dimensions,
            vector_name=f"document vector {index}",
        )
        for index, vector in enumerate(vectors)
    )


def _validate_vector(
    raw_vector: Sequence[float],
    *,
    expected_dimensions: int,
    vector_name: str,
) -> EmbeddingVector:
    if isinstance(raw_vector, (str, bytes)):
        raise EmbeddingResultError(f"{vector_name} must be a numeric sequence")
    try:
        values = tuple(raw_vector)
    except TypeError as exc:
        raise EmbeddingResultError(f"{vector_name} must be iterable") from exc

    if len(values) != expected_dimensions:
        raise EmbeddingResultError(
            f"{vector_name} has {len(values)} dimensions; expected {expected_dimensions}"
        )

    normalized_values: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise EmbeddingResultError(f"{vector_name}[{index}] must be numeric")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise EmbeddingResultError(f"{vector_name}[{index}] must be numeric") from exc
        if not isfinite(numeric_value):
            raise EmbeddingResultError(f"{vector_name}[{index}] must be finite")
        normalized_values.append(numeric_value)
    return tuple(normalized_values)
