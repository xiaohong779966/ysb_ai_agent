"""Static validation for Docker and Docker Compose foundation files."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_compose_defines_only_backend_service() -> None:
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text("utf-8"))

    assert set(compose["services"]) == {"backend"}
    backend = compose["services"]["backend"]
    assert backend["build"] == {
        "context": ".",
        "dockerfile": "backend/Dockerfile",
    }
    assert backend["env_file"] == [".env"]
    assert "${PORT:-8000}:8000" in backend["ports"]
    assert backend["environment"]["LOG_FORMAT"] == "json"


def test_compose_healthcheck_targets_versioned_endpoint() -> None:
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text("utf-8"))
    healthcheck = compose["services"]["backend"]["healthcheck"]
    command = " ".join(healthcheck["test"])

    assert healthcheck["test"][0] == "CMD"
    assert "/api/v1/health" in command
    assert "127.0.0.1:8000" in command
    assert healthcheck["retries"] == 3


def test_dockerfile_uses_non_root_python_service() -> None:
    dockerfile = (PROJECT_ROOT / "backend" / "Dockerfile").read_text("utf-8")

    assert dockerfile.startswith("FROM python:3.12-slim")
    assert "USER app" in dockerfile
    assert "backend.app.main:app" in dockerfile
    assert 'EXPOSE 8000' in dockerfile
    assert "requirements.txt" in dockerfile
