import os

# Must be set before CUDA initializes to reduce allocator fragmentation on 12GB cards.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import jobs, router
from app.core.config import ROOT_DIR, get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    jobs.start()
    yield
    await jobs.stop()


app = FastAPI(
    title="LTX-Video 2B API",
    description="Demo backend: text-to-video or image-to-video with LTX-Video 2B distilled.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

static_dir = ROOT_DIR / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def demo_page():
    from fastapi.responses import FileResponse

    index = static_dir / "index.html"
    if not index.is_file():
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/docs")
    return FileResponse(index)
