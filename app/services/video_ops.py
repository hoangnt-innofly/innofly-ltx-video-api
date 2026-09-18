from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np


def concat_videos(
    paths: list[str | Path],
    output_path: str | Path,
    *,
    frame_rate: int,
    interlude: np.ndarray | None = None,
    interlude_frames: int = 0,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output_path,
        fps=frame_rate,
        format="FFMPEG",
        codec="libx264",
        pixelformat="yuv420p",
    ) as writer:
        for index, path in enumerate(paths):
            reader = imageio.get_reader(path)
            try:
                for frame in reader:
                    writer.append_data(np.asarray(frame))
            finally:
                reader.close()
            if interlude is not None and interlude_frames > 0 and index < len(paths) - 1:
                for _ in range(interlude_frames):
                    writer.append_data(interlude)
    return output_path
