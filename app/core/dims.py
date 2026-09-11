def align_resolution(value: int, multiple: int = 32) -> int:
    if value < multiple:
        return multiple
    return (value // multiple) * multiple


def align_frames(value: int) -> int:
    """LTX-Video requires frame count of the form 8n + 1 (9, 17, 25, 49, 81...)."""
    if value < 9:
        return 9
    return ((value - 1) // 8) * 8 + 1


def clamp_for_low_vram(
    width: int,
    height: int,
    num_frames: int,
    *,
    max_width: int = 512,
    max_height: int = 320,
    max_frames: int = 49,
) -> tuple[int, int, int]:
    """Keep aspect ratio while fitting a 12GB-class GPU budget."""
    num_frames = min(align_frames(num_frames), max_frames)
    width = align_resolution(width)
    height = align_resolution(height)
    if width <= max_width and height <= max_height:
        return width, height, num_frames
    scale = min(max_width / width, max_height / height)
    return (
        align_resolution(max(32, int(width * scale))),
        align_resolution(max(32, int(height * scale))),
        num_frames,
    )
