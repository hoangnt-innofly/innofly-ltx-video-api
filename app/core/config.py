from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    public_base_url: str = "http://127.0.0.1:8000"

    ltx_mock: bool = False
    ltx_pipeline_config: str = "configs/ltxv-2b-0.9.8-distilled-lowvram.yaml"

    ltx_default_width: int = 512
    ltx_default_height: int = 320
    ltx_default_num_frames: int = 33
    ltx_default_frame_rate: int = 24
    ltx_default_seed: int = 42
    ltx_negative_prompt: str = (
        "worst quality, inconsistent motion, blurry, jittery, distorted, "
        "morphing, warped face, extra fingers, plastic skin, oversharpened"
    )
    ltx_default_image_cond_noise_scale: float = 0.15

    # Portrait-friendly defaults for the two-image outfit-change screen.
    ltx_outfit_width: int = 320
    ltx_outfit_height: int = 512
    ltx_outfit_num_frames: int = 49
    ltx_outfit_frame_rate: int = 16
    ltx_outfit_image_cond_noise_scale: float = 0.22
    ltx_outfit_direction: str = "right"

    max_upload_mb: int = 20
    job_ttl_seconds: int = 86400
    # 0 = no artificial cap. Do not set 10/11 on an 11.6GiB card.
    ltx_max_gpu_memory_gb: float = 0.0
    ltx_cpu_offload: bool = True

    @property
    def pipeline_config_path(self) -> Path:
        path = Path(self.ltx_pipeline_config)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path

    @property
    def upload_dir(self) -> Path:
        return ROOT_DIR / "storage" / "uploads"

    @property
    def video_dir(self) -> Path:
        return ROOT_DIR / "storage" / "videos"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.video_dir.mkdir(parents=True, exist_ok=True)
    return settings
