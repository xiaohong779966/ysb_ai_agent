"""Version 1 API router."""

from fastapi import APIRouter

from backend.app.api.v1.endpoints.health import router as health_router
from backend.app.api.v1.endpoints.knowledge import router as knowledge_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router)
api_v1_router.include_router(knowledge_router)
