"""Provider-neutral embedding contracts used by vector knowledge modules."""

from backend.app.rag.embeddings.base import (
    EmbeddingBatch,
    EmbeddingProvider,
    EmbeddingVector,
)
from backend.app.rag.embeddings.service import EmbeddingService

__all__ = [
    "EmbeddingBatch",
    "EmbeddingProvider",
    "EmbeddingService",
    "EmbeddingVector",
]
