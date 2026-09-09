"""Shared pytest fixtures for deterministic environment-based tests."""

import pytest

from backend.app.knowledge.models import KnowledgeValidationIssue

SETTINGS_ENVIRONMENT_VARIABLES = (
    "APP_NAME",
    "APP_VERSION",
    "APP_ENV",
    "DEBUG",
    "API_V1_PREFIX",
    "HOST",
    "PORT",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "KNOWLEDGE_UPLOAD_MAX_MB",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DEVICE",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_NORMALIZE",
    "VECTOR_STORE_PROVIDER",
    "CHROMA_PERSIST_DIRECTORY",
    "CHROMA_COLLECTION_NAME",
    "CHROMA_DISTANCE_METRIC",
    "CHROMA_QUERY_LIMIT",
)


@pytest.fixture(autouse=True)
def isolate_application_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent host environment variables from leaking into test settings."""
    for variable in SETTINGS_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


class Helpers:
    """Small factories that keep expected domain objects readable in tests."""

    @staticmethod
    def issue(
        row_number: int,
        field: str,
        code: str,
        message: str,
    ) -> KnowledgeValidationIssue:
        return KnowledgeValidationIssue(
            row_number=row_number,
            field=field,
            code=code,
            message=message,
        )


@pytest.fixture
def helpers() -> Helpers:
    return Helpers()
