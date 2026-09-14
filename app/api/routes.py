from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.core.dims import align_frames, align_resolution
from app.models.schemas import HealthResponse, JobResponse
from app.services.jobs import JobService

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

router = APIRouter()
settings = get_settings()
jobs = JobService(settings)


def public_base(request: Request) -> str:
    configured = settings.public_base_url.rstrip("/")
    if configured:
        return configured
    return str(request.base_url).rstrip("/")


def to_response(request: Request, job) -> JobResponse:
    return JobResponse.model_validate(job.to_public(public_base(request)))


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cuda_available, device_name = jobs.cuda_info()
    return HealthResponse(
        mock=settings.ltx_mock,
        cuda_available=cuda_available,
        device_name=device_name,
        pipeline_ready=jobs.pipeline_ready(),
        defaults={
            "width": settings.ltx_default_width,
            "height": settings.ltx_default_height,
            "num_frames": settings.ltx_default_num_frames,
            "frame_rate": settings.ltx_default_frame_rate,
            "seed": settings.ltx_default_seed,
            "model": "ltxv-2b-0.9.8-distilled",
            "pipeline_config": settings.ltx_pipeline_config,
            "max_gpu_memory_gb": settings.ltx_max_gpu_memory_gb,
            "cpu_offload": settings.ltx_cpu_offload,
        },
    )


@router.post("/api/v1/generate", response_model=JobResponse)
async def generate(
    request: Request,
    image: Annotated[UploadFile, File(description="Conditioning image for Image-to-Video")],
    prompt: Annotated[str, Form(min_length=1)],
    width: Annotated[Optional[int], Form()] = None,
    height: Annotated[Optional[int], Form()] = None,
    num_frames: Annotated[Optional[int], Form()] = None,
    frame_rate: Annotated[Optional[int], Form()] = None,
    seed: Annotated[Optional[int], Form()] = None,
    negative_prompt: Annotated[Optional[str], Form()] = None,
    wait: Annotated[bool, Form()] = True,
) -> JobResponse:
    """Upload an image + prompt. By default waits until the mp4 is ready and returns video_url."""
    job = await _enqueue(image, prompt, width, height, num_frames, frame_rate, seed, negative_prompt)
    if wait:
        try:
            job = await jobs.wait(job.id, timeout=1200)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise HTTPException(status_code=504, detail="Generation timed out") from exc
        if job.status == "failed":
            raise HTTPException(status_code=500, detail=job.error or "Generation failed")
    return to_response(request, job)


@router.post("/api/v1/jobs", response_model=JobResponse, status_code=202)
async def create_job(
    request: Request,
    image: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form(min_length=1)],
    width: Annotated[Optional[int], Form()] = None,
    height: Annotated[Optional[int], Form()] = None,
    num_frames: Annotated[Optional[int], Form()] = None,
    frame_rate: Annotated[Optional[int], Form()] = None,
    seed: Annotated[Optional[int], Form()] = None,
    negative_prompt: Annotated[Optional[str], Form()] = None,
) -> JobResponse:
    """Queue a job and return immediately. Poll GET /api/v1/jobs/{job_id} for video_url."""
    job = await _enqueue(image, prompt, width, height, num_frames, frame_rate, seed, negative_prompt)
    return to_response(request, job)


@router.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request) -> JobResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return to_response(request, job)


@router.get("/media/videos/{filename}")
def get_video(filename: str):
    path = settings.video_dir / Path(filename).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


async def _enqueue(
    image: UploadFile,
    prompt: str,
    width: int | None,
    height: int | None,
    num_frames: int | None,
    frame_rate: int | None,
    seed: int | None,
    negative_prompt: str | None,
):
    suffix = Path(image.filename or "input.jpg").suffix.lower()
    content_type = (image.content_type or "").lower()
    if suffix not in ALLOWED_SUFFIXES and content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Image must be jpg, png, or webp")

    data = await image.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {settings.max_upload_mb} MB",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")

    if width is not None:
        width = align_resolution(width)
    if height is not None:
        height = align_resolution(height)
    if num_frames is not None:
        num_frames = align_frames(num_frames)

    dest = settings.upload_dir / f"{uuid4().hex}{suffix or '.jpg'}"
    dest.write_bytes(data)
    return jobs.create_job(
        prompt=prompt,
        image_path=dest,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        seed=seed,
        negative_prompt=negative_prompt,
    )
