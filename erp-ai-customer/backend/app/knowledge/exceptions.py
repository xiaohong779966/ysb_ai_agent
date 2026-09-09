"""Exceptions raised when an Excel knowledge workbook cannot be parsed."""


class KnowledgeWorkbookError(ValueError):
    """Base error for workbook-level parsing failures."""


class UnsupportedWorkbookTypeError(KnowledgeWorkbookError):
    """Raised when a file is not an XLSX workbook."""


class InvalidWorkbookError(KnowledgeWorkbookError):
    """Raised when an XLSX file is corrupt or unreadable."""


class InvalidKnowledgeHeadersError(KnowledgeWorkbookError):
    """Raised when the worksheet does not follow the required two-column schema."""
