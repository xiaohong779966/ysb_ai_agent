"""API response models for Excel knowledge-base validation and preview."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class KnowledgeRecordResponse(BaseModel):
    """One valid question-answer row returned in the upload preview."""

    model_config = ConfigDict(frozen=True)

    row_number: int
    question: str
    answer: str


class KnowledgeValidationIssueResponse(BaseModel):
    """One actionable row validation issue returned to the client."""

    model_config = ConfigDict(frozen=True)

    row_number: int | None
    field: Literal["workbook", "row", "question", "answer"]
    code: str
    message: str


class KnowledgeUploadResponse(BaseModel):
    """Validation summary and bounded preview for an uploaded workbook."""

    model_config = ConfigDict(frozen=True)

    file_name: str
    sheet_name: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    is_valid: bool
    preview_count: int
    preview_truncated: bool
    records: list[KnowledgeRecordResponse]
    issues: list[KnowledgeValidationIssueResponse]
