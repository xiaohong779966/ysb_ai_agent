"""Response models exposed by the service foundation endpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ServiceInfoResponse(BaseModel):
    """Basic service metadata returned from the root endpoint."""

    model_config = ConfigDict(frozen=True)

    service: str
    version: str
    docs_url: str


class HealthResponse(BaseModel):
    """Health status returned by the versioned health endpoint."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    service: str
    version: str
    environment: str
