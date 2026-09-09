"""Tests for validated Chroma vector-store configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.config.settings import PROJECT_ROOT, Settings


def test_vector_store_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.vector_store_provider == "chroma"
    assert settings.chroma_persist_directory == Path("data/chroma")
    assert settings.chroma_persist_path == (PROJECT_ROOT / "data/chroma").resolve()
    assert settings.chroma_collection_name == "erp_faq"
    assert settings.chroma_distance_metric == "cosine"
    assert settings.chroma_query_limit == 5


def test_vector_store_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persist_directory = Path("D:/erp-vector-test/vectors")
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "CHROMA")
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(persist_directory))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "ERP_FAQ_TEST")
    monkeypatch.setenv("CHROMA_DISTANCE_METRIC", "IP")
    monkeypatch.setenv("CHROMA_QUERY_LIMIT", "12")

    settings = Settings(_env_file=None)

    assert settings.vector_store_provider == "chroma"
    assert settings.chroma_persist_path == persist_directory.resolve()
    assert settings.chroma_collection_name == "erp_faq_test"
    assert settings.chroma_distance_metric == "ip"
    assert settings.chroma_query_limit == 12


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VECTOR_STORE_PROVIDER", "milvus"),
        ("CHROMA_PERSIST_DIRECTORY", "  "),
        ("CHROMA_COLLECTION_NAME", "ab"),
        ("CHROMA_COLLECTION_NAME", "contains spaces"),
        ("CHROMA_COLLECTION_NAME", "invalid..name"),
        ("CHROMA_COLLECTION_NAME", "127.0.0.1"),
        ("CHROMA_DISTANCE_METRIC", "euclidean"),
        ("CHROMA_QUERY_LIMIT", "0"),
        ("CHROMA_QUERY_LIMIT", "101"),
    ],
)
def test_invalid_vector_store_settings_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
