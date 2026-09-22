from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

WalkDirection = str

NEGATIVE_EXTRA = (
    "person frozen in place, standing still in the center, morphing clothes, "
    "two people, clone, extra limbs, warped face, sliding teleport, camera pan, "
    "head cropped, missing face, headless, face cut off, torso only, clothes only, "
    "zoomed in on body, close-up of outfit, no head"
)

WALK_OUT_PROMPTS = {
    "right": (
        "Photorealistic vertical 9:16 full-body studio video, locked tripod camera, "
        "light gray seamless backdrop, first frames match the reference photo exactly: "
        "the same face, head, hair and full body stay in frame, no zoom, no crop. "
        "Then that same person turns and walks naturally to the right, face still visible "
        "until they leave the right edge, casual gait, arms swinging, empty studio remains, "
        "no cut, no clothes change, no close-up of the torso"
    ),
    "left": (
        "Photorealistic vertical 9:16 full-body studio video, locked tripod camera, "
        "light gray seamless backdrop, first frames match the reference photo exactly: "
        "the same face, head, hair and full body stay in frame, no zoom, no crop. "
        "Then that same person turns and walks naturally to the left, face still visible "
        "until they leave the left edge, casual gait, arms swinging, empty studio remains, "
        "no cut, no clothes change, no close-up of the torso"
    ),
}

WALK_IN_PROMPTS = {
    "right": (
        "Photorealistic vertical 9:16 full-body studio video, locked tripod camera, "
        "light gray seamless backdrop, the same person enters from the right edge walking "
        "left toward the center, face and head visible the whole time, natural gait, "
        "then stops in the middle facing the camera, full body, same identity and lighting, "
        "no morphing, no zoom into clothes"
    ),
    "left": (
        "Photorealistic vertical 9:16 full-body studio video, locked tripod camera, "
        "light gray seamless backdrop, the same person enters from the left edge walking "
        "right toward the center, face and head visible the whole time, natural gait, "
        "then stops in the middle facing the camera, full body, same identity and lighting, "
        "no morphing, no zoom into clothes"
    ),
}


def normalize_direction(value: str | None) -> str:
    direction = (value or "right").strip().lower()
    if direction not in {"left", "right"}:
        raise ValueError("direction must be left or right")
    return direction


def walk_out_prompt(direction: str) -> str:
    return WALK_OUT_PROMPTS[normalize_direction(direction)]


def walk_in_prompt(direction: str) -> str:
    return WALK_IN_PROMPTS[normalize_direction(direction)]


def resolve_prompt(override: str | None, fallback: str) -> str:
    cleaned = (override or "").strip()
    return cleaned or fallback


def prompts_catalog() -> dict:
    return {
        "default_direction": "right",
        "directions": ["right", "left"],
        "prompts": {
            "right": {
                "walk_out_prompt": WALK_OUT_PROMPTS["right"],
                "walk_in_prompt": WALK_IN_PROMPTS["right"],
            },
            "left": {
                "walk_out_prompt": WALK_OUT_PROMPTS["left"],
                "walk_in_prompt": WALK_IN_PROMPTS["left"],
            },
        },
    }


def sample_background(image: Image.Image) -> tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w, _ = arr.shape
    bh, bw = max(1, h // 40), max(1, w // 40)
    patches = np.concatenate(
        [
            arr[:bh, :bw].reshape(-1, 3),
            arr[:bh, w - bw :].reshape(-1, 3),
            arr[h - bh :, :bw].reshape(-1, 3),
            arr[h - bh :, w - bw :].reshape(-1, 3),
        ],
        axis=0,
    )
    return tuple(int(x) for x in patches.mean(axis=0))


def prepare_start_frame(
    image_path: str | Path,
    output_path: str | Path,
    *,
    width: int,
    height: int,
) -> Path:
    """Letterbox the photo onto the job canvas so I2V does not center-crop the head."""
    source = Image.open(image_path).convert("RGB")
    fitted = fit_on_canvas(source, width, height)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fitted.save(output_path, quality=95)
    return output_path


def prepare_enter_frame(
    image_path: str | Path,
    output_path: str | Path,
    *,
    direction: str,
    width: int,
    height: int,
    shift_ratio: float = 0.46,
) -> Path:
    """Place the person on the entrance edge so I2V can walk them back to center."""
    direction = normalize_direction(direction)
    source = Image.open(image_path).convert("RGB")
    fitted = fit_on_canvas(source, width, height)
    bg = sample_background(fitted)
    canvas = Image.new("RGB", (width, height), bg)
    dx = int(width * shift_ratio)
    if direction == "right":
        canvas.paste(fitted, (dx, 0))
    else:
        canvas.paste(fitted, (-dx, 0))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    return output_path


def empty_studio_frame(image_path: str | Path, width: int, height: int) -> np.ndarray:
    source = Image.open(image_path).convert("RGB")
    fitted = fit_on_canvas(source, width, height)
    bg = sample_background(fitted)
    return np.full((height, width, 3), bg, dtype=np.uint8)


def fit_on_canvas(image: Image.Image, width: int, height: int) -> Image.Image:
    src_w, src_h = image.size
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    bg = sample_background(image)
    canvas = Image.new("RGB", (width, height), bg)
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas
