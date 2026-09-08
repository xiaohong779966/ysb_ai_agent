"""Tests for validated application settings."""

import pytest
from pydantic import ValidationError

from backend.app.config.settings import Settings


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "ERP AI Customer Service"
    assert settings.app_env == "local"
    assert settings.debug is False
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.port == 8000
    assert settings.log_level == "INFO"
    assert settings.log_format == "text"


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("API_V1_PREFIX", "service/v1/")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_FORMAT", "JSON")

    settings = Settings(_env_file=None)

    assert settings.app_env == "testing"
    assert settings.debug is True
    assert settings.api_v1_prefix == "/service/v1"
    assert settings.port == 9000
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "json"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("APP_ENV", "unknown"),
        ("LOG_LEVEL", "verbose"),
        ("LOG_FORMAT", "xml"),
        ("API_V1_PREFIX", "/"),
    ],
)
def test_invalid_settings_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
