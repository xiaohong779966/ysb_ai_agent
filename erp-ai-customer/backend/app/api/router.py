"""Top-level API router assembly."""

from fastapi import APIRouter

from backend.app.api.root import router as root_router
from backend.app.api.v1.router import api_v1_router


def build_api_router(api_v1_prefix: str) -> APIRouter:
    """Build the root and versioned API routes for an application instance."""
    router = APIRouter()
    router.include_router(root_router)
    router.include_router(api_v1_router, prefix=api_v1_prefix)
    return router
