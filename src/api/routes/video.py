"""
POST /video/frame  — process a single raw frame (for streaming integrations)
POST /video/file   — upload a full video; get back a violation summary JSON
GET  /video/log    — download the current violations log
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()


@router.post("/frame")
async def process_frame(request: Request, file: UploadFile = File(...)):
    """
    Process a single JPEG/PNG frame.
    Intended for real-time streaming: client sends frames one-by-one.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG/PNG frame).")

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode frame.")

    pipeline = request.state.pipeline
    result = pipeline.process_frame(frame, annotate=False)

    return {
        "frame_number": result.frame_number,
        "persons": len(result.persons),
        "new_violations": result.new_violations,
        "total_violations": pipeline.violation_tracker.total_violations,
    }


@router.post("/file")
async def process_video_file(request: Request, file: UploadFile = File(...)):
    """
    Upload a video file; process every frame; return violation summary.
    NOTE: blocking — use a task queue (Celery / ARQ) for production.
    """
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise HTTPException(400, "Could not open video file.")

    pipeline = request.state.pipeline
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pipeline.process_frame(frame, annotate=False)
        total_frames += 1

    cap.release()
    Path(tmp_path).unlink(missing_ok=True)

    log_path = pipeline.violation_tracker.export_log()

    return {
        "total_frames_processed": total_frames,
        "total_violations": pipeline.violation_tracker.total_violations,
        "violation_ids": list(pipeline.violation_tracker._confirmed),
        "log_file": str(log_path),
    }


@router.get("/log")
async def get_violation_log(request: Request):
    """Download the violations JSON log."""
    pipeline = request.state.pipeline
    log_path = pipeline.violation_tracker.export_log()
    if not log_path.exists():
        raise HTTPException(404, "No log file found.")
    return FileResponse(str(log_path), media_type="application/json", filename="violations_log.json")
