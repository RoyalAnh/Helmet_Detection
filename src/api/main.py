"""
FastAPI application entry point.
Mounts the three route groups: health, image, video.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from configs.settings import settings
from src.pipeline import HelmetPipeline
from src.api.routes import health, image, video


# ── Shared app state ────────────────────────────────────────────────────────
pipeline: HelmetPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info(f"Loading model from {settings.model_path} on {settings.device}")
    pipeline = HelmetPipeline(settings)
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Helmet Violation Detection API",
    description="Detect persons without helmets in images or video frames.",
    version="1.0.0",
    lifespan=lifespan,
)

# Inject pipeline into request state via middleware
@app.middleware("http")
async def attach_pipeline(request, call_next):
    request.state.pipeline = pipeline
    return await call_next(request)


app.include_router(health.router, tags=["Health"])
app.include_router(image.router, prefix="/detect", tags=["Detection"])
app.include_router(video.router, prefix="/video",  tags=["Video"])
