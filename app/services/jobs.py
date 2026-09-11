from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.core.config import Settings
from app.core.dims import align_frames, align_resolution
from app.services.mock_engine import generate_placeholder_video

logger = logging.getLogger("ltx-api")

JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class Job:
    id: str
    prompt: str
    image_path: Path
    video_path: Path
    width: int
    height: int
    num_frames: int
    frame_rate: int
    seed: int
    negative_prompt: str
    status: JobStatus = "queued"
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def to_public(self, base_url: str) -> dict:
        video_url = None
        if self.status == "succeeded":
            video_url = f"{base_url.rstrip('/')}/media/videos/{self.video_path.name}"
        return {
            "job_id": self.id,
            "status": self.status,
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "seed": self.seed,
            "video_url": video_url,
            "error": self.error,
        }


class JobService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._engine = None
        self._engine_lock = threading.Lock()
        self._worker_task: asyncio.Task | None = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop(), name="ltx-worker")

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def create_job(
        self,
        *,
        prompt: str,
        image_path: Path,
        width: int | None,
        height: int | None,
        num_frames: int | None,
        frame_rate: int | None,
        seed: int | None,
        negative_prompt: str | None,
    ) -> Job:
        job_id = uuid.uuid4().hex
        width = align_resolution(width or self.settings.ltx_default_width)
        height = align_resolution(height or self.settings.ltx_default_height)
        num_frames = align_frames(num_frames or self.settings.ltx_default_num_frames)
        job = Job(
            id=job_id,
            prompt=prompt.strip(),
            image_path=image_path,
            video_path=self.settings.video_dir / f"{job_id}.mp4",
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate or self.settings.ltx_default_frame_rate,
            seed=seed if seed is not None else self.settings.ltx_default_seed,
            negative_prompt=negative_prompt or self.settings.ltx_negative_prompt,
        )
        self.jobs[job_id] = job
        self._queue.put_nowait(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    async def wait(self, job_id: str, timeout: float = 900) -> Job:
        job = self.jobs[job_id]
        await asyncio.wait_for(job._done.wait(), timeout=timeout)
        return job

    def pipeline_ready(self) -> bool:
        if self.settings.ltx_mock:
            return True
        return bool(self._engine and getattr(self._engine, "ready", False))

    def cuda_info(self) -> tuple[bool, str | None]:
        try:
            import torch

            if torch.cuda.is_available():
                return True, torch.cuda.get_device_name(0)
        except Exception:
            pass
        return False, None

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self.jobs[job_id]
            job.status = "running"
            try:
                await asyncio.to_thread(self._run_job, job)
                job.status = "succeeded"
            except Exception as exc:
                logger.exception("Job %s failed", job_id)
                job.status = "failed"
                job.error = str(exc)
            finally:
                job.finished_at = datetime.now(timezone.utc)
                job._done.set()
                self._queue.task_done()

    def _run_job(self, job: Job) -> None:
        if self.settings.ltx_mock:
            generate_placeholder_video(
                job.image_path,
                job.video_path,
                width=job.width,
                height=job.height,
                num_frames=job.num_frames,
                frame_rate=job.frame_rate,
            )
            return

        engine = self._get_engine()
        engine.generate_i2v(
            image_path=job.image_path,
            prompt=job.prompt,
            output_path=job.video_path,
            height=job.height,
            width=job.width,
            num_frames=job.num_frames,
            frame_rate=job.frame_rate,
            seed=job.seed,
            negative_prompt=job.negative_prompt,
        )

    def _get_engine(self):
        with self._engine_lock:
            if self._engine is None:
                from app.services.ltx_engine import LTXEngine

                self._engine = LTXEngine(self.settings.pipeline_config_path)
                self._engine.load()
            return self._engine
