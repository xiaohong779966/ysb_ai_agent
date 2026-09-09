"""Immutable result models for ERP knowledge retrieval."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeRetrievalHit:
    """One validated ERP FAQ match ordered by vector-store distance."""

    record_id: str
    question: str
    answer: str
    document: str
    source_file: str
    sheet_name: str
    row_number: int
    distance: float


@dataclass(frozen=True, slots=True)
class KnowledgeRetrievalResult:
    """Summary and ordered hits for one ERP knowledge search."""

    query: str
    limit: int
    collection_name: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    hits: tuple[KnowledgeRetrievalHit, ...]

    @property
    def total_hits(self) -> int:
        """Return the number of matches produced by this search."""
        return len(self.hits)
