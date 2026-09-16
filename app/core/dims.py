def align_resolution(value: int, multiple: int = 32) -> int:
    if value < multiple:
        return multiple
    return (value // multiple) * multiple


def align_frames(value: int) -> int:
    """LTX-Video requires frame count of the form 8n + 1 (9, 17, 25, 49, 81...)."""
    if value < 9:
        return 9
    return ((value - 1) // 8) * 8 + 1
