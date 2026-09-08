"""FastAPI application factory for the ERP AI customer service backend."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.router import build_api_router
from backend.app.config.logging import configure_logging
from backend.app.config.settings import Settings, get_settings
from backend.app.middleware.request_context import RequestContextMiddleware

_logger = logging.getLogger("erp_customer_service.lifecycle")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create a configured FastAPI application instance."""
    active_settings = settings or get_settings()
    configure_logging(active_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        _logger.info(
            "Application started",
            extra={"environment": active_settings.app_env},
        )
        yield
        _logger.info("Application stopped")

    application = FastAPI(
        title=active_settings.app_name,
        version=active_settings.app_version,
        debug=active_settings.debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    application.state.settings = active_settings
    application.add_middleware(RequestContextMiddleware)
    application.include_router(build_api_router(active_settings.api_v1_prefix))
    return application


app = create_app()
