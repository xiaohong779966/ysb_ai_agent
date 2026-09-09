"""Typed errors raised by provider-neutral vector storage."""


class VectorStoreError(Exception):
    """Base exception for vector-store failures."""


class VectorStoreConfigurationError(VectorStoreError, ValueError):
    """Raised when a vector store cannot be configured safely."""


class VectorStoreInputError(VectorStoreError, ValueError):
    """Raised when records, embeddings, filters, or identifiers are invalid."""


class VectorStoreDependencyError(VectorStoreError, RuntimeError):
    """Raised when a vector database runtime dependency is unavailable."""


class VectorStoreOperationError(VectorStoreError, RuntimeError):
    """Raised when a vector database operation fails."""


class VectorStoreResultError(VectorStoreError, RuntimeError):
    """Raised when a vector database returns a malformed result."""
