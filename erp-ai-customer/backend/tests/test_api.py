"""API contract tests for the FastAPI foundation."""

from fastapi.testclient import TestClient

from backend.app.config.settings import Settings
from backend.app.main import create_app
from backend.app.middleware.request_context import REQUEST_ID_HEADER


def build_client() -> TestClient:
    settings = Settings(
        app_name="ERP Test Service",
        app_version="1.2.3",
        app_env="testing",
        log_level="CRITICAL",
        _env_file=None,
    )
    return TestClient(create_app(settings))


def test_root_endpoint_returns_service_metadata() -> None:
    with build_client() as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "ERP Test Service",
        "version": "1.2.3",
        "docs_url": "/docs",
    }
    assert response.headers[REQUEST_ID_HEADER]


def test_health_endpoint_returns_validated_status() -> None:
    with build_client() as client:
        response = client.get(
            "/api/v1/health",
            headers={REQUEST_ID_HEADER: "health-check-001"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "ERP Test Service",
        "version": "1.2.3",
        "environment": "testing",
    }
    assert response.headers[REQUEST_ID_HEADER] == "health-check-001"


def test_openapi_and_swagger_are_available() -> None:
    with build_client() as client:
        openapi_response = client.get("/openapi.json")
        docs_response = client.get("/docs")

    assert openapi_response.status_code == 200
    assert docs_response.status_code == 200
    assert openapi_response.json()["info"] == {
        "title": "ERP Test Service",
        "version": "1.2.3",
    }
    assert "/api/v1/health" in openapi_response.json()["paths"]


def test_application_factory_returns_independent_instances() -> None:
    settings = Settings(app_env="testing", log_level="CRITICAL", _env_file=None)

    first_app = create_app(settings)
    second_app = create_app(settings)

    assert first_app is not second_app
    assert first_app.state.settings is settings
    assert second_app.state.settings is settings
