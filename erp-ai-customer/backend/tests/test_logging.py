"""Tests for central logging configuration."""

import json
import logging

from backend.app.config.logging import JsonFormatter, RequestIdFilter, configure_logging
from backend.app.config.settings import Settings
from backend.app.middleware.request_context import reset_request_id, set_request_id


def test_json_formatter_contains_required_fields() -> None:
    token = set_request_id("request-123")
    try:
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="测试消息",
            args=(),
            exc_info=None,
        )
        record.method = "GET"
        record.path = "/health"
        record.status_code = 200
        record.duration_ms = 12.5
        RequestIdFilter().filter(record)

        payload = json.loads(JsonFormatter().format(record))
    finally:
        reset_request_id(token)

    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["message"] == "测试消息"
    assert payload["request_id"] == "request-123"
    assert payload["method"] == "GET"
    assert payload["path"] == "/health"
    assert payload["status_code"] == 200
    assert payload["duration_ms"] == 12.5
    assert payload["timestamp"].endswith("+00:00")


def test_configure_logging_is_idempotent() -> None:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    try:
        configure_logging(Settings(log_format="json", _env_file=None))
        configure_logging(Settings(log_format="json", _env_file=None))

        service_handlers = [
            handler
            for handler in root_logger.handlers
            if getattr(handler, "_erp_customer_service_handler", False)
        ]
        assert len(service_handlers) == 1
        assert isinstance(service_handlers[0].formatter, JsonFormatter)
    finally:
        for handler in list(root_logger.handlers):
            if getattr(handler, "_erp_customer_service_handler", False):
                root_logger.removeHandler(handler)
                handler.close()
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
