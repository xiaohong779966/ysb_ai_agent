"""Central logging configuration for the API service."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from backend.app.config.settings import Settings
from backend.app.middleware.request_context import get_request_id

_HANDLER_MARKER = "_erp_customer_service_handler"
_REQUEST_FIELDS = ("method", "path", "status_code", "duration_ms")


class RequestIdFilter(logging.Filter):
    """Attach the active request ID to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for field in _REQUEST_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(settings: Settings) -> None:
    """Configure the root logger without adding duplicate service handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    for handler in list(root_logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root_logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler()
    setattr(handler, _HANDLER_MARKER, True)
    handler.setLevel(settings.log_level)
    handler.addFilter(RequestIdFilter())

    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | "
                "request_id=%(request_id)s | %(message)s"
            )
        )

    root_logger.addHandler(handler)
