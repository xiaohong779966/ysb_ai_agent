"""Domain models returned by the Excel knowledge-base parser."""

from dataclasses import dataclass
from typing import Literal

KnowledgeField = Literal["workbook", "row", "question", "answer"]


@dataclass(frozen=True, slots=True)
class KnowledgeRecord:
    """One normalized and validated ERP question-answer pair."""

    row_number: int
    question: str
    answer: str


@dataclass(frozen=True, slots=True)
class KnowledgeValidationIssue:
    """One actionable validation problem found in a workbook."""

    row_number: int | None
    field: KnowledgeField
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class KnowledgeParseResult:
    """Validated records and row-level issues from one worksheet."""

    sheet_name: str
    total_rows: int
    records: tuple[KnowledgeRecord, ...]
    issues: tuple[KnowledgeValidationIssue, ...]

    @property
    def valid_rows(self) -> int:
        return len(self.records)

    @property
    def invalid_rows(self) -> int:
        return len({issue.row_number for issue in self.issues if issue.row_number is not None})

    @property
    def is_valid(self) -> bool:
        return not self.issues
