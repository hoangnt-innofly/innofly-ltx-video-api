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
    ltx_pipeline_config: str = "configs/ltxv-2b-0.9.8-distilled.yaml"

    ltx_default_width: int = 704
    ltx_default_height: int = 480
    ltx_default_num_frames: int = 49
    ltx_default_frame_rate: int = 24
    ltx_default_seed: int = 42
    ltx_negative_prompt: str = (
        "worst quality, inconsistent motion, blurry, jittery, distorted"
    )

    max_upload_mb: int = 20
    job_ttl_seconds: int = 86400

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
