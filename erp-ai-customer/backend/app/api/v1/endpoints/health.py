"""Health check endpoint for local and container orchestration probes."""

from fastapi import APIRouter, Request

from backend.app.config.settings import Settings
from backend.app.schemas.service import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Service health check")
async def health_check(request: Request) -> HealthResponse:
    settings: Settings = request.app.state.settings
    return HealthResponse(
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )
