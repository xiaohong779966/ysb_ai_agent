"""Embed user questions and retrieve compatible ERP FAQ vectors."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

from backend.app.rag.embeddings.exceptions import EmbeddingError
from backend.app.rag.embeddings.service import EmbeddingService
from backend.app.rag.retrieval.exceptions import (
    KnowledgeRetrievalConfigurationError,
    KnowledgeRetrievalEmbeddingError,
    KnowledgeRetrievalInputError,
    KnowledgeRetrievalResultError,
    KnowledgeRetrievalStorageError,
)
from backend.app.rag.retrieval.models import (
    KnowledgeRetrievalHit,
    KnowledgeRetrievalResult,
)
from backend.app.rag.vector_store.base import VectorSearchResult, VectorStore
from backend.app.rag.vector_store.exceptions import VectorStoreError

DOCUMENT_TYPE = "erp_faq"
DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 100
DOCUMENT_QUESTION_PREFIX = "问题："
DOCUMENT_ANSWER_MARKER = "\n答案："


class KnowledgeRetrievalService:
    """Coordinate query embedding and provider-neutral vector search."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        default_limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._default_limit = _validate_limit(
            default_limit,
            error_type=KnowledgeRetrievalConfigurationError,
            input_name="default limit",
        )

    @property
    def embedding_service(self) -> EmbeddingService:
        """Return the validated embedding boundary used for queries."""
        return self._embedding_service

    @property
    def vector_store(self) -> VectorStore:
        """Return the provider-neutral vector store used for retrieval."""
        return self._vector_store

    @property
    def default_limit(self) -> int:
        """Return the maximum matches requested when callers omit a limit."""
        return self._default_limit

    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
    ) -> KnowledgeRetrievalResult:
        """Embed one normalized question and return compatible ERP FAQ matches."""
        normalized_query = _normalize_query(query)
        effective_limit = self.default_limit
        if limit is not None:
            effective_limit = _validate_limit(
                limit,
                error_type=KnowledgeRetrievalInputError,
                input_name="limit",
            )

        try:
            query_embedding = self.embedding_service.embed_query(normalized_query)
        except EmbeddingError as exc:
            raise KnowledgeRetrievalEmbeddingError(
                "Failed to create the knowledge-search query embedding"
            ) from exc
        except Exception as exc:
            raise KnowledgeRetrievalEmbeddingError(
                "Unexpected failure while creating the knowledge-search query embedding"
            ) from exc

        provider = self.embedding_service.provider
        compatibility_filter = _build_compatibility_filter(
            embedding_provider=provider.provider_name,
            embedding_model=provider.model_name,
            embedding_dimensions=self.embedding_service.dimensions,
        )
        try:
            raw_hits = self.vector_store.query(
                query_embedding,
                limit=effective_limit,
                where=compatibility_filter,
            )
        except VectorStoreError as exc:
            raise KnowledgeRetrievalStorageError(
                f"Failed to query vector store collection "
                f"'{self.vector_store.collection_name}'"
            ) from exc
        except Exception as exc:
            raise KnowledgeRetrievalStorageError(
                f"Unexpected vector store failure for collection "
                f"'{self.vector_store.collection_name}'"
            ) from exc

        hits = _normalize_hits(
            raw_hits,
            limit=effective_limit,
            embedding_provider=provider.provider_name,
            embedding_model=provider.model_name,
            embedding_dimensions=self.embedding_service.dimensions,
        )
        return KnowledgeRetrievalResult(
            query=normalized_query,
            limit=effective_limit,
            collection_name=self.vector_store.collection_name,
            embedding_provider=provider.provider_name,
            embedding_model=provider.model_name,
            embedding_dimensions=self.embedding_service.dimensions,
            hits=hits,
        )


def _validate_limit(
    value: object,
    *,
    error_type: type[KnowledgeRetrievalConfigurationError]
    | type[KnowledgeRetrievalInputError],
    input_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f"{input_name} must be an integer")
    if not 1 <= value <= MAX_SEARCH_LIMIT:
        raise error_type(f"{input_name} must be between 1 and {MAX_SEARCH_LIMIT}")
    return value


def _normalize_query(query: object) -> str:
    if not isinstance(query, str):
        raise KnowledgeRetrievalInputError("query must be a string")
    normalized = query.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise KnowledgeRetrievalInputError("query must not be blank")
    return normalized


def _build_compatibility_filter(
    *,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int,
) -> dict[str, object]:
    return {
        "$and": [
            {"document_type": DOCUMENT_TYPE},
            {"embedding_provider": embedding_provider},
            {"embedding_model": embedding_model},
            {"embedding_dimensions": embedding_dimensions},
        ]
    }


def _normalize_hits(
    raw_hits: object,
    *,
    limit: int,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int,
) -> tuple[KnowledgeRetrievalHit, ...]:
    if isinstance(raw_hits, (str, bytes)):
        raise KnowledgeRetrievalResultError(
            "Vector store returned an invalid knowledge-search result sequence"
        )
    try:
        hits = tuple(raw_hits)  # type: ignore[arg-type]
    except TypeError as exc:
        raise KnowledgeRetrievalResultError(
            "Vector store returned a non-iterable knowledge-search result"
        ) from exc
    if len(hits) > limit:
        raise KnowledgeRetrievalResultError(
            f"Vector store returned {len(hits)} hits for a limit of {limit}"
        )

    normalized_hits: list[KnowledgeRetrievalHit] = []
    seen_ids: set[str] = set()
    for index, raw_hit in enumerate(hits):
        if not isinstance(raw_hit, VectorSearchResult):
            raise KnowledgeRetrievalResultError(
                f"Vector store result hits[{index}] must be a VectorSearchResult"
            )
        hit = _normalize_hit(
            raw_hit,
            index=index,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
        )
        if hit.record_id in seen_ids:
            raise KnowledgeRetrievalResultError(
                f"Vector store result contains duplicate record ID '{hit.record_id}'"
            )
        seen_ids.add(hit.record_id)
        normalized_hits.append(hit)
    return tuple(normalized_hits)


def _normalize_hit(
    raw_hit: VectorSearchResult,
    *,
    index: int,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int,
) -> KnowledgeRetrievalHit:
    record_id = _require_text(raw_hit.id, input_name=f"hits[{index}].id")
    document = _require_text(raw_hit.document, input_name=f"hits[{index}].document")
    if not isinstance(raw_hit.metadata, Mapping):
        raise KnowledgeRetrievalResultError(f"hits[{index}].metadata must be a mapping")
    metadata = raw_hit.metadata

    document_type = _require_metadata_text(metadata, "document_type", index=index)
    if document_type != DOCUMENT_TYPE:
        raise KnowledgeRetrievalResultError(
            f"hits[{index}] is not an ERP FAQ knowledge document"
        )
    source_file = _require_metadata_text(metadata, "source_file", index=index)
    if not source_file.lower().endswith(".xlsx"):
        raise KnowledgeRetrievalResultError(
            f"hits[{index}].metadata['source_file'] must use the .xlsx extension"
        )
    sheet_name = _require_metadata_text(metadata, "sheet_name", index=index)
    question = _require_metadata_text(metadata, "question", index=index)
    stored_provider = _require_metadata_text(metadata, "embedding_provider", index=index)
    stored_model = _require_metadata_text(metadata, "embedding_model", index=index)
    row_number = _require_metadata_integer(metadata, "row_number", index=index, minimum=2)
    stored_dimensions = _require_metadata_integer(
        metadata,
        "embedding_dimensions",
        index=index,
        minimum=1,
    )
    if (
        stored_provider != embedding_provider
        or stored_model != embedding_model
        or stored_dimensions != embedding_dimensions
    ):
        raise KnowledgeRetrievalResultError(
            f"hits[{index}] was created with an incompatible embedding configuration"
        )

    expected_prefix = f"{DOCUMENT_QUESTION_PREFIX}{question}{DOCUMENT_ANSWER_MARKER}"
    if not document.startswith(expected_prefix):
        raise KnowledgeRetrievalResultError(
            f"hits[{index}].document does not match its question metadata"
        )
    answer = document[len(expected_prefix) :].strip()
    if not answer:
        raise KnowledgeRetrievalResultError(f"hits[{index}].document contains a blank answer")

    if isinstance(raw_hit.distance, bool):
        raise KnowledgeRetrievalResultError(f"hits[{index}].distance must be numeric")
    try:
        distance = float(raw_hit.distance)
    except (TypeError, ValueError) as exc:
        raise KnowledgeRetrievalResultError(
            f"hits[{index}].distance must be numeric"
        ) from exc
    if not isfinite(distance):
        raise KnowledgeRetrievalResultError(f"hits[{index}].distance must be finite")

    return KnowledgeRetrievalHit(
        record_id=record_id,
        question=question,
        answer=answer,
        document=document,
        source_file=source_file,
        sheet_name=sheet_name,
        row_number=row_number,
        distance=distance,
    )


def _require_metadata_text(
    metadata: Mapping[str, object],
    key: str,
    *,
    index: int,
) -> str:
    return _require_text(
        metadata.get(key),
        input_name=f"hits[{index}].metadata['{key}']",
    )


def _require_metadata_integer(
    metadata: Mapping[str, object],
    key: str,
    *,
    index: int,
    minimum: int,
) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise KnowledgeRetrievalResultError(
            f"hits[{index}].metadata['{key}'] must be an integer "
            f"greater than or equal to {minimum}"
        )
    return value


def _require_text(value: object, *, input_name: str) -> str:
    if not isinstance(value, str):
        raise KnowledgeRetrievalResultError(f"{input_name} must be a string")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise KnowledgeRetrievalResultError(f"{input_name} must not be blank")
    return normalized
