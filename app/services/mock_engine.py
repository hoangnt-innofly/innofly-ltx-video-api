from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
from PIL import Image

from app.core.outfit import empty_studio_frame, fit_on_canvas, normalize_direction


def generate_placeholder_video(
    image_path: str | Path | None,
    output_path: str | Path,
    *,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: int,
) -> Path:
    """Still-image mp4 so the API contract can be tested without a GPU."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image_path and Path(image_path).is_file():
        image = Image.open(image_path).convert("RGB")
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        base = np.asarray(image, dtype=np.uint8)
    else:
        rng = np.random.default_rng(width * 1000 + height)
        base = rng.integers(48, 196, size=(height, width, 3), dtype=np.uint8)

    with imageio.get_writer(output_path, fps=frame_rate, format="FFMPEG", codec="libx264") as writer:
        for i in range(num_frames):
            pulse = 1.0 - 0.08 * abs(((i % 24) / 12.0) - 1.0)
            frame = np.clip(base.astype(np.float32) * pulse, 0, 255).astype(np.uint8)
            writer.append_data(frame)

    return output_path


def generate_outfit_placeholder(
    image_before: str | Path,
    image_after: str | Path,
    output_path: str | Path,
    *,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: int,
    direction: str = "right",
) -> Path:
    """Slide image 1 off-screen, then slide image 2 back in (no GPU)."""
    direction = normalize_direction(direction)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    before = np.asarray(
        fit_on_canvas(Image.open(image_before).convert("RGB"), width, height),
        dtype=np.uint8,
    )
    after = np.asarray(
        fit_on_canvas(Image.open(image_after).convert("RGB"), width, height),
        dtype=np.uint8,
    )
    empty = empty_studio_frame(image_before, width, height)
    hold = max(6, num_frames // 8)
    sign = 1 if direction == "right" else -1

    with imageio.get_writer(
        output_path,
        fps=frame_rate,
        format="FFMPEG",
        codec="libx264",
        pixelformat="yuv420p",
    ) as writer:
        for i in range(num_frames):
            t = i / max(1, num_frames - 1)
            writer.append_data(_shift_frame(before, empty, int(sign * t * width)))
        for _ in range(hold):
            writer.append_data(empty)
        for i in range(num_frames):
            t = i / max(1, num_frames - 1)
            writer.append_data(_shift_frame(after, empty, int(sign * (1.0 - t) * width)))
    return output_path


def _shift_frame(subject: np.ndarray, background: np.ndarray, dx: int) -> np.ndarray:
    height, width, _ = background.shape
    canvas = background.copy()
    if dx >= width or dx <= -width:
        return canvas
    if dx >= 0:
        src_x0, src_x1 = 0, width - dx
        dst_x0, dst_x1 = dx, width
    else:
        src_x0, src_x1 = -dx, width
        dst_x0, dst_x1 = 0, width + dx
    canvas[:, dst_x0:dst_x1] = subject[:, src_x0:src_x1]
    return canvas
