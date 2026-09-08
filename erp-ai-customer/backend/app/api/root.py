"""Root service metadata endpoint."""

from fastapi import APIRouter, Request

from backend.app.config.settings import Settings
from backend.app.schemas.service import ServiceInfoResponse

router = APIRouter(tags=["service"])


@router.get("/", response_model=ServiceInfoResponse, summary="Service information")
async def service_information(request: Request) -> ServiceInfoResponse:
    settings: Settings = request.app.state.settings
    return ServiceInfoResponse(
        service=settings.app_name,
        version=settings.app_version,
        docs_url=request.app.docs_url or "",
    )
