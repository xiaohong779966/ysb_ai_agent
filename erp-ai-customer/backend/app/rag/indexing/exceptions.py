"""Typed failures raised while converting ERP knowledge into stored vectors."""


class KnowledgeIndexError(Exception):
    """Base exception for knowledge indexing failures."""


class KnowledgeIndexInputError(KnowledgeIndexError, ValueError):
    """Raised when an indexing request contains malformed domain data."""


class KnowledgeIndexValidationError(KnowledgeIndexError, ValueError):
    """Raised when a workbook has validation issues and cannot be indexed atomically."""


class KnowledgeIndexEmbeddingError(KnowledgeIndexError, RuntimeError):
    """Raised when document embedding fails during an indexing operation."""


class KnowledgeIndexStorageError(KnowledgeIndexError, RuntimeError):
    """Raised when generated vectors cannot be persisted."""


class KnowledgeIndexResultError(KnowledgeIndexError, RuntimeError):
    """Raised when a dependency reports an inconsistent indexing result."""
