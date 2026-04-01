"""
Wraps deep_sort_realtime.DeepSort.
Converts person detections → tracked detections with stable track_id.
"""
from __future__ import annotations

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.schemas import Detection
from src.utils.geometry import compute_iou


class SortTracker:
    def __init__(self, max_age: int, n_init: int) -> None:
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            embedder="mobilenet",
            half=True, # False 
            bgr=True,
        )

    def update(
        self,
        persons: list[Detection],
        frame: np.ndarray,
    ) -> list[Detection]:
        """
        Feed current person detections into the tracker.
        Returns the same detections with `track_id` populated
        (only confirmed tracks are returned).
        """
        raw = [
            ([p.bbox[0], p.bbox[1], p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]],
             p.confidence,
             p.class_id)
            for p in persons
        ]
        tracks = self._tracker.update_tracks(raw, frame=frame)

        tracked: list[Detection] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            # Find the original detection closest to this track
            best = self._match_person(ltrb, persons)
            if best is None:
                continue
            best.track_id = track.track_id
            best.bbox = ltrb   # use the Kalman-filtered bbox
            tracked.append(best)

        return tracked

    # ------------------------------------------------------------------
    @staticmethod
    def _match_person(
        ltrb: list[float],
        persons: list[Detection],
    ) -> Detection | None:
        best_iou, best = 0.0, None
        for p in persons:
            iou = compute_iou(ltrb, p.bbox)
            if iou > best_iou:
                best_iou, best = iou, p
        return best
