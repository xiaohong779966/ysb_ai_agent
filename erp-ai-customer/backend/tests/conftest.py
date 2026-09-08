"""Shared pytest fixtures for deterministic environment-based tests."""

import pytest

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
)


@pytest.fixture(autouse=True)
def isolate_application_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent host environment variables from leaking into test settings."""
    for variable in SETTINGS_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
