"""Request correlation context and HTTP access logging middleware."""

import logging
from contextvars import ContextVar, Token
from time import perf_counter
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")
_logger = logging.getLogger("erp_customer_service.request")


def get_request_id() -> str:
    """Return the request ID associated with the current execution context."""
    return _REQUEST_ID.get()


def set_request_id(request_id: str) -> Token[str]:
    """Set a request ID and return the context token needed to restore it."""
    return _REQUEST_ID.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    """Restore the prior request ID context."""
    _REQUEST_ID.reset(token)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Add request correlation IDs and emit one structured access log entry."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        incoming_request_id = request.headers.get(REQUEST_ID_HEADER, "").strip()
        request_id = incoming_request_id[:128] or str(uuid4())
        token = set_request_id(request_id)
        started_at = perf_counter()
        status_code = 500
        response_completed = False

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            response_completed = True
            return response
        except Exception:
            _logger.exception(
                "Unhandled request error",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                },
            )
            raise
        finally:
            if response_completed:
                _logger.info(
                    "Request completed",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": status_code,
                        "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    },
                )
            reset_request_id(token)
