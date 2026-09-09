"""Typed failures raised by the provider-neutral embedding layer."""


class EmbeddingError(Exception):
    """Base exception for embedding contract failures."""


class EmbeddingConfigurationError(EmbeddingError, ValueError):
    """Raised when provider metadata cannot define a valid embedding space."""


class EmbeddingInputError(EmbeddingError, ValueError):
    """Raised when text supplied for embedding is missing or invalid."""


class EmbeddingResultError(EmbeddingError, RuntimeError):
    """Raised when a provider returns malformed or incompatible vectors."""
