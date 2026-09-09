"""Provider-neutral type contracts for text embedding implementations."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

type EmbeddingVector = tuple[float, ...]
type EmbeddingBatch = tuple[EmbeddingVector, ...]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract implemented by local or remote text embedding providers."""

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier used by configuration."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model identifier used to create the vectors."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the number of elements in every generated vector."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed document texts while preserving their input order."""
        ...

    def embed_query(self, text: str) -> Sequence[float]:
        """Embed a search query using any provider-specific query behavior."""
        ...
