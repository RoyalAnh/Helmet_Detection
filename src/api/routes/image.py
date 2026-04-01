"""
POST /detect/image
------------------
Upload a single image; receive back:
  - JSON body with detections + any new violations
  - The annotated image as a JPEG (base64 in JSON, or raw bytes via /image/raw)
"""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import Response

router = APIRouter()


@router.post("/image")
async def detect_image(request: Request, file: UploadFile = File(...)):
    """Detect helmet violations in a single uploaded image. Returns JSON."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    contents = await file.read()
    frame = _decode_image(contents)
    if frame is None:
        raise HTTPException(400, "Could not decode image.")

    pipeline = request.state.pipeline
    result = pipeline.process_frame(frame, annotate=True)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    img_b64 = base64.b64encode(buf).decode()

    return {
        "frame_number": result.frame_number,
        "fps": round(result.fps, 1),
        "persons_detected": len(result.persons),
        "new_violations": result.new_violations,
        "total_violations": pipeline.violation_tracker.total_violations,
        "annotated_image_b64": img_b64,
    }


@router.post("/image/raw", response_class=Response)
async def detect_image_raw(request: Request, file: UploadFile = File(...)):
    """Same as /image but returns the annotated image as raw JPEG bytes."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    contents = await file.read()
    frame = _decode_image(contents)
    if frame is None:
        raise HTTPException(400, "Could not decode image.")

    pipeline = request.state.pipeline
    pipeline.process_frame(frame, annotate=True)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
