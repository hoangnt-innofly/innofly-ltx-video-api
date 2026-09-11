"""Download LTX-Video 2B distilled weights into ./models."""

from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FILES = [
    "ltxv-2b-0.9.8-distilled.safetensors",
    "ltxv-spatial-upscaler-0.9.8.safetensors",
]


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        path = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename=filename,
            local_dir=str(MODELS),
        )
        print(path)


if __name__ == "__main__":
    main()
