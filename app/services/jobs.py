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
from app.core.outfit import (
    NEGATIVE_EXTRA,
    empty_studio_frame,
    normalize_direction,
    prepare_enter_frame,
    resolve_prompt,
    walk_in_prompt,
    walk_out_prompt,
)
from app.services.mock_engine import generate_outfit_placeholder, generate_placeholder_video
from app.services.video_ops import concat_videos

logger = logging.getLogger("ltx-api")

JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class Job:
    id: str
    prompt: str
    image_path: Path | None
    video_path: Path
    width: int
    height: int
    num_frames: int
    frame_rate: int
    seed: int
    negative_prompt: str
    image_cond_noise_scale: float = 0.15
    image2_path: Path | None = None
    mode: str = "i2v"
    direction: str | None = None
    walk_out_prompt: str | None = None
    walk_in_prompt: str | None = None
    status: JobStatus = "queued"
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def to_public(self, base_url: str) -> dict:
        video_url = None
        if self.status == "succeeded":
            video_url = f"{base_url.rstrip('/')}/media/videos/{self.video_path.name}"
        if self.mode == "outfit_change":
            mode = "outfit_change"
        elif self.image_path is not None:
            mode = "i2v"
        else:
            mode = "t2v"
        return {
            "job_id": self.id,
            "status": self.status,
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "frame_rate": self.frame_rate,
            "seed": self.seed,
            "mode": mode,
            "direction": self.direction,
            "walk_out_prompt": self.walk_out_prompt,
            "walk_in_prompt": self.walk_in_prompt,
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
        image_path: Path | None,
        width: int | None,
        height: int | None,
        num_frames: int | None,
        frame_rate: int | None,
        seed: int | None,
        negative_prompt: str | None,
        image_cond_noise_scale: float | None = None,
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
            image_cond_noise_scale=(
                image_cond_noise_scale
                if image_cond_noise_scale is not None
                else self.settings.ltx_default_image_cond_noise_scale
            ),
            mode="i2v" if image_path is not None else "t2v",
        )
        self.jobs[job_id] = job
        self._queue.put_nowait(job_id)
        return job

    def create_outfit_job(
        self,
        *,
        image_before: Path,
        image_after: Path,
        direction: str | None = None,
        walk_out_prompt_text: str | None = None,
        walk_in_prompt_text: str | None = None,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
        frame_rate: int | None = None,
        seed: int | None = None,
        image_cond_noise_scale: float | None = None,
    ) -> Job:
        job_id = uuid.uuid4().hex
        walk = normalize_direction(direction or self.settings.ltx_outfit_direction)
        out_prompt = resolve_prompt(walk_out_prompt_text, walk_out_prompt(walk))
        in_prompt = resolve_prompt(walk_in_prompt_text, walk_in_prompt(walk))
        width = align_resolution(width or self.settings.ltx_outfit_width)
        height = align_resolution(height or self.settings.ltx_outfit_height)
        num_frames = align_frames(num_frames or self.settings.ltx_outfit_num_frames)
        negative = f"{self.settings.ltx_negative_prompt}, {NEGATIVE_EXTRA}"
        job = Job(
            id=job_id,
            prompt=out_prompt,
            image_path=image_before,
            image2_path=image_after,
            video_path=self.settings.video_dir / f"{job_id}.mp4",
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate or self.settings.ltx_outfit_frame_rate,
            seed=seed if seed is not None else self.settings.ltx_default_seed,
            negative_prompt=negative,
            image_cond_noise_scale=(
                image_cond_noise_scale
                if image_cond_noise_scale is not None
                else self.settings.ltx_outfit_image_cond_noise_scale
            ),
            mode="outfit_change",
            direction=walk,
            walk_out_prompt=out_prompt,
            walk_in_prompt=in_prompt,
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

    @staticmethod
    def _free_cuda() -> None:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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
                self._free_cuda()
            finally:
                job.finished_at = datetime.now(timezone.utc)
                job._done.set()
                self._queue.task_done()

    def _run_job(self, job: Job) -> None:
        if job.mode == "outfit_change":
            self._run_outfit_job(job)
            return

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
        engine.generate(
            prompt=job.prompt,
            output_path=job.video_path,
            height=job.height,
            width=job.width,
            num_frames=job.num_frames,
            frame_rate=job.frame_rate,
            seed=job.seed,
            negative_prompt=job.negative_prompt,
            image_path=job.image_path,
            image_cond_noise_scale=job.image_cond_noise_scale,
        )

    def _run_outfit_job(self, job: Job) -> None:
        if job.image_path is None or job.image2_path is None:
            raise ValueError("Outfit change needs both images")
        direction = normalize_direction(job.direction)

        if self.settings.ltx_mock:
            generate_outfit_placeholder(
                job.image_path,
                job.image2_path,
                job.video_path,
                width=job.width,
                height=job.height,
                num_frames=job.num_frames,
                frame_rate=job.frame_rate,
                direction=direction,
            )
            return

        enter_path = self.settings.upload_dir / f"{job.id}_enter.jpg"
        prepare_enter_frame(
            job.image2_path,
            enter_path,
            direction=direction,
            width=job.width,
            height=job.height,
        )
        clip_out = self.settings.video_dir / f"{job.id}_out.mp4"
        clip_in = self.settings.video_dir / f"{job.id}_in.mp4"
        engine = self._get_engine()
        engine.generate(
            prompt=job.walk_out_prompt or walk_out_prompt(direction),
            output_path=clip_out,
            height=job.height,
            width=job.width,
            num_frames=job.num_frames,
            frame_rate=job.frame_rate,
            seed=job.seed,
            negative_prompt=job.negative_prompt,
            image_path=job.image_path,
            image_cond_noise_scale=job.image_cond_noise_scale,
            single_scale=True,
        )
        self._free_cuda()
        engine.generate(
            prompt=job.walk_in_prompt or walk_in_prompt(direction),
            output_path=clip_in,
            height=job.height,
            width=job.width,
            num_frames=job.num_frames,
            frame_rate=job.frame_rate,
            seed=job.seed + 1,
            negative_prompt=job.negative_prompt,
            image_path=enter_path,
            image_cond_noise_scale=job.image_cond_noise_scale,
            single_scale=True,
        )
        hold = max(6, job.frame_rate // 3)
        concat_videos(
            [clip_out, clip_in],
            job.video_path,
            frame_rate=job.frame_rate,
            interlude=empty_studio_frame(job.image_path, job.width, job.height),
            interlude_frames=hold,
        )

    def _get_engine(self):
        with self._engine_lock:
            if self._engine is None:
                from app.services.ltx_engine import LTXEngine

                self._engine = LTXEngine(
                    self.settings.pipeline_config_path,
                    max_gpu_memory_gb=self.settings.ltx_max_gpu_memory_gb,
                    cpu_offload=self.settings.ltx_cpu_offload,
                )
                try:
                    self._engine.load()
                except Exception:
                    self._engine = None
                    raise
            return self._engine
