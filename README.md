# LTX-Video 2B API

FastAPI demo backend for local **Image-to-Video** with [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video) **`ltxv-2b-0.9.8-distilled`** (RTX 3060 12GB).

Upload an image + text prompt. Submit returns a `video_url`.

## Defaults (3060 12GB)

| Setting | Value |
| --- | --- |
| Model | `ltxv-2b-0.9.8-distilled` (single-scale low-VRAM config) |
| Resolution | `512×320` (divisible by 32) |
| Frames | `49` (`8n + 1`) |
| Prompt enhancer | off (saves VRAM) |

`704×480` multi-scale OOMs on 12GB. After a first run succeeds, you can try `640×384` then `704×480`. On 16GB+ switch `LTX_PIPELINE_CONFIG` to `configs/ltxv-2b-0.9.8-distilled.yaml`.

## API

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/api/v1/generate` | Upload image + prompt, **wait**, return `video_url` |
| `POST` | `/api/v1/jobs` | Same upload, return immediately (`202`) |
| `GET` | `/api/v1/jobs/{job_id}` | Poll until `status=succeeded` and `video_url` is set |
| `GET` | `/media/videos/{file}` | Stream the mp4 |
| `GET` | `/health` | CUDA / mock / defaults |
| `GET` | `/docs` | Swagger |

Multipart fields: `image`, `prompt`. Optional: `width`, `height`, `num_frames`, `frame_rate`, `seed`, `negative_prompt`.

Example (sync demo — response includes `video_url`):

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/generate" `
  -F "image=@input.jpg" `
  -F "prompt=A young anime boy slowly turns his head and smiles, cinematic animation" `
  -F "width=512" `
  -F "height=320" `
  -F "num_frames=49" `
  -F "seed=42"
```

```json
{
  "job_id": "...",
  "status": "succeeded",
  "prompt": "...",
  "width": 512,
  "height": 320,
  "num_frames": 49,
  "seed": 42,
  "video_url": "http://127.0.0.1:8000/media/videos/<job_id>.mp4",
  "error": null
}
```

Jobs are serialized on one GPU worker so concurrent requests do not OOM the 3060.

## Setup (Windows + Python 3.11)

```powershell
cd ltx-video-api
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

Use **Python 3.11** for real GPU inference (LTX-Video / CUDA torch). A newer interpreter is fine only for `LTX_MOCK=1`.

### A. API contract only (no GPU)

In `.env` set `LTX_MOCK=1`, then:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000 — upload any jpg, get a placeholder mp4 URL.

### B. Real LTX-Video 2B (RTX 3060)

Install CUDA PyTorch, then the official package:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install "git+https://github.com/Lightricks/LTX-Video.git#egg=ltx-video[inference]"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python scripts/download_models.py
```

Copy `.env.example` to `.env` on the GPU server. On 12GB cards keep `LTX_PIPELINE_CONFIG=configs/ltxv-2b-0.9.8-distilled-lowvram.yaml`, `LTX_MAX_GPU_MEMORY_GB=0`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Generate at 512×320 × 49 — do not use 704×480 or the multi-scale yaml. First request downloads remaining Hugging Face deps (T5 text encoder) if they are not cached, then loads the pipeline once and reuses it.

Weights used:

- `ltxv-2b-0.9.8-distilled.safetensors`
- `ltxv-spatial-upscaler-0.9.8.safetensors` (multi-scale distilled config)

## Project layout

```
app/            FastAPI app, job queue, LTX engine
configs/        2B distilled yaml (prompt enhancement disabled)
scripts/        weight download helper
static/         tiny upload demo page
storage/        uploads + generated mp4s
```
