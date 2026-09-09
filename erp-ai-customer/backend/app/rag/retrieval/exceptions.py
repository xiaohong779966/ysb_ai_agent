"""Typed failures raised while retrieving ERP knowledge vectors."""


class KnowledgeRetrievalError(Exception):
    """Base exception for ERP knowledge retrieval failures."""


class KnowledgeRetrievalConfigurationError(KnowledgeRetrievalError, ValueError):
    """Raised when the retrieval service is configured with invalid limits."""


class KnowledgeRetrievalInputError(KnowledgeRetrievalError, ValueError):
    """Raised when a retrieval request contains an invalid query or limit."""


class KnowledgeRetrievalEmbeddingError(KnowledgeRetrievalError, RuntimeError):
    """Raised when the search query cannot be embedded."""


class KnowledgeRetrievalStorageError(KnowledgeRetrievalError, RuntimeError):
    """Raised when the vector store cannot execute a similarity query."""


class KnowledgeRetrievalResultError(KnowledgeRetrievalError, RuntimeError):
    """Raised when a vector-store result is incompatible with ERP FAQ data."""
