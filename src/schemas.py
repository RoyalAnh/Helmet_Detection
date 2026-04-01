"""
Shared data classes used across detector, tracker, and API layers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    """Raw output from the YOLO model for a single object."""
    bbox: list[float]        # [x1, y1, x2, y2] — absolute pixel coords
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None


@dataclass
class ViolationRecord:
    """A confirmed helmet violation tied to one tracked person."""
    track_id: int
    timestamp: str           # ISO-8601
    frame_number: int
    bbox: list[float]
    confidence: float
    snapshot_path: Optional[str] = None


@dataclass
class FrameResult:
    """Everything the pipeline returns for a single processed frame."""
    frame_number: int
    persons: list[Detection] = field(default_factory=list)
    helmets: list[Detection] = field(default_factory=list)
    new_violations: list[int] = field(default_factory=list)   # track_ids
    fps: float = 0.0
