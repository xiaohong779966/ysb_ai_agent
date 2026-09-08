"""Tests for request correlation middleware."""

import logging
from uuid import UUID

import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from backend.app.middleware.request_context import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    get_request_id,
)


def build_test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/request-id")
    async def request_id_endpoint() -> dict[str, str]:
        return {"request_id": get_request_id()}

    return app


def test_middleware_generates_request_id() -> None:
    response = TestClient(build_test_app()).get("/request-id")

    response_request_id = response.headers[REQUEST_ID_HEADER]
    UUID(response_request_id)
    assert response.json() == {"request_id": response_request_id}


def test_middleware_preserves_incoming_request_id() -> None:
    response = TestClient(build_test_app()).get(
        "/request-id",
        headers={REQUEST_ID_HEADER: "customer-request-42"},
    )

    assert response.headers[REQUEST_ID_HEADER] == "customer-request-42"
    assert response.json() == {"request_id": "customer-request-42"}


def test_middleware_logs_completed_500_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/failure")
    async def failure_endpoint() -> Response:
        return Response(status_code=500)

    with caplog.at_level(logging.INFO, logger="erp_customer_service.request"):
        response = TestClient(app).get("/failure")

    assert response.status_code == 500
    completion_records = [
        record for record in caplog.records if record.getMessage() == "Request completed"
    ]
    assert len(completion_records) == 1
    assert completion_records[0].status_code == 500
