"""Documentation and environment-template acceptance tests."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ENVIRONMENT_VARIABLES = {
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
}


def test_environment_example_documents_all_foundation_settings() -> None:
    lines = (PROJECT_ROOT / ".env.example").read_text("utf-8").splitlines()
    documented_variables = {
        line.split("=", maxsplit=1)[0]
        for line in lines
        if line and not line.startswith("#")
    }

    assert documented_variables == REQUIRED_ENVIRONMENT_VARIABLES


def test_readme_contains_local_docker_and_api_instructions() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text("utf-8")

    required_content = (
        "Python 3.12",
        "python -m uvicorn backend.app.main:app",
        "python -m pytest -q",
        "python -m ruff check backend",
        "docker compose build",
        "/api/v1/health",
        "/docs",
        "未安装或未启用 Docker CLI",
    )
    for item in required_content:
        assert item in readme

