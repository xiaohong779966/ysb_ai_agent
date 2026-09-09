"""Immutable result models for ERP knowledge indexing."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeIndexResult:
    """Summary of one successfully completed atomic knowledge import."""

    source_name: str
    sheet_name: str
    total_rows: int
    indexed_records: int
    collection_name: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    record_ids: tuple[str, ...]
