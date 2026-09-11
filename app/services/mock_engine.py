from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
from PIL import Image


def generate_placeholder_video(
    image_path: str | Path,
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

    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    base = np.asarray(image, dtype=np.uint8)

    with imageio.get_writer(output_path, fps=frame_rate, format="FFMPEG", codec="libx264") as writer:
        for i in range(num_frames):
            pulse = 1.0 - 0.08 * abs(((i % 24) / 12.0) - 1.0)
            frame = np.clip(base.astype(np.float32) * pulse, 0, 255).astype(np.uint8)
            writer.append_data(frame)

    return output_path
