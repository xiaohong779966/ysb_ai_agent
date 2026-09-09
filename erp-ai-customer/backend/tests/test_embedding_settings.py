"""Tests for validated embedding configuration."""

import pytest
from pydantic import ValidationError

from backend.app.config.settings import Settings


def test_embedding_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.embedding_provider == "sentence_transformers"
    assert settings.embedding_model == "BAAI/bge-small-zh-v1.5"
    assert settings.embedding_device == "auto"
    assert settings.embedding_batch_size == 32
    assert settings.embedding_normalize is True


def test_embedding_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "SENTENCE_TRANSFORMERS")
    monkeypatch.setenv("EMBEDDING_MODEL", "  local/custom-model  ")
    monkeypatch.setenv("EMBEDDING_DEVICE", "CPU")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "16")
    monkeypatch.setenv("EMBEDDING_NORMALIZE", "false")

    settings = Settings(_env_file=None)

    assert settings.embedding_provider == "sentence_transformers"
    assert settings.embedding_model == "local/custom-model"
    assert settings.embedding_device == "cpu"
    assert settings.embedding_batch_size == 16
    assert settings.embedding_normalize is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EMBEDDING_PROVIDER", "unsupported"),
        ("EMBEDDING_MODEL", "  "),
        ("EMBEDDING_DEVICE", "tpu"),
        ("EMBEDDING_BATCH_SIZE", "0"),
        ("EMBEDDING_BATCH_SIZE", "513"),
        ("EMBEDDING_NORMALIZE", "sometimes"),
    ],
)
def test_invalid_embedding_settings_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
