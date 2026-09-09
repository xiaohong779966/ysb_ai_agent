"""Provider-neutral contracts for persistent vector storage."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from backend.app.rag.embeddings.base import EmbeddingVector

type VectorMetadataValue = str | int | float | bool
type VectorMetadata = Mapping[str, VectorMetadataValue]


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One embedded document ready to be persisted in a vector collection."""

    id: str
    document: str
    embedding: EmbeddingVector
    metadata: VectorMetadata


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """One vector search hit with its source document and distance."""

    id: str
    document: str
    metadata: VectorMetadata
    distance: float


@runtime_checkable
class VectorStore(Protocol):
    """Contract implemented by persistent vector database adapters."""

    @property
    def provider_name(self) -> str:
        """Return the stable vector-store provider identifier."""
        ...

    @property
    def collection_name(self) -> str:
        """Return the collection used for ERP knowledge vectors."""
        ...

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Insert or replace records and return the processed record count."""
        ...

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int | None = None,
        where: Mapping[str, object] | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Return nearest records ordered by the provider's distance."""
        ...

    def delete(self, ids: Sequence[str]) -> None:
        """Delete records by identifier."""
        ...

    def count(self) -> int:
        """Return the number of records in the collection."""
        ...
