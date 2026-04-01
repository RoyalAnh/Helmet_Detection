"""
Tracks per-identity helmet status across frames.

A violation is only "confirmed" when the same track_id has been
seen without a helmet for `confirm_frames` consecutive frames.
This prevents false positives from momentary detection gaps.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.schemas import ViolationRecord


class ViolationTracker:
    def __init__(
        self,
        confirm_frames: int,
        output_dir: str,
        save_snapshots: bool = True,
    ) -> None:
        self._confirm = confirm_frames
        self._output_dir = Path(output_dir)
        self._save_snapshots = save_snapshots

        # Rolling window: track_id -> deque of bool (True = violation frame)
        self._history: dict[int, deque[bool]] = defaultdict(
            lambda: deque(maxlen=confirm_frames)
        )
        self._confirmed: set[int] = set()
        self.records: list[ViolationRecord] = []

    # ------------------------------------------------------------------
    def update(
        self,
        track_id: int,
        has_helmet: bool,
        frame_number: int,
        bbox: list[float],
        confidence: float,
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Update state for one track.
        Returns True only the first time a violation is confirmed.
        """
        self._history[track_id].append(not has_helmet)

        window = list(self._history[track_id])
        is_violation = len(window) == self._confirm and all(window)

        if is_violation and track_id not in self._confirmed:
            self._confirmed.add(track_id)
            snapshot = self._save_snapshot(track_id, frame, bbox, frame_number) if (
                self._save_snapshots and frame is not None
            ) else None
            self.records.append(ViolationRecord(
                track_id=track_id,
                timestamp=datetime.now().isoformat(),
                frame_number=frame_number,
                bbox=bbox,
                confidence=confidence,
                snapshot_path=snapshot,
            ))
            return True

        return False

    # ------------------------------------------------------------------
    def export_log(self, filename: str = "violations_log.json") -> Path:
        path = self._output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "track_id": r.track_id,
                "timestamp": r.timestamp,
                "frame_number": r.frame_number,
                "bbox": [round(v, 1) for v in r.bbox],
                "confidence": round(r.confidence, 3),
                "snapshot": r.snapshot_path,
            }
            for r in self.records
        ]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return path

    @property
    def total_violations(self) -> int:
        return len(self._confirmed)

    # ------------------------------------------------------------------
    def _save_snapshot(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: list[float],
        frame_number: int,
    ) -> str:
        save_dir = self._output_dir / "snapshots"
        save_dir.mkdir(parents=True, exist_ok=True)

        h, w = frame.shape[:2]
        pad = 20
        x1, y1, x2, y2 = (max(0, int(v)) for v in bbox)
        crop = frame[
            max(0, y1 - pad): min(h, y2 + pad),
            max(0, x1 - pad): min(w, x2 + pad),
        ]
        path = str(save_dir / f"track{track_id}_frame{frame_number}.jpg")
        cv2.imwrite(path, crop)
        return path
