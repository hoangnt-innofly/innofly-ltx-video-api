from typing import Any

from pydantic import BaseModel, Field


class JobResponse(BaseModel):
    job_id: str
    status: str
    prompt: str
    width: int
    height: int
    num_frames: int
    seed: int
    video_url: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    mock: bool
    cuda_available: bool
    device_name: str | None = None
    pipeline_ready: bool
    defaults: dict[str, Any] = Field(default_factory=dict)
