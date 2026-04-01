"""
HelmetPipeline
==============
Composes YOLODetector → HelmetAssociator → SortTracker → ViolationTracker
into a single `process_frame()` call.

Usage
-----
    pipeline = HelmetPipeline()
    result = pipeline.process_frame(bgr_frame)
"""
from __future__ import annotations

import numpy as np
import time
from collections import deque # FPS history

from configs.settings import settings as default_settings, Settings
from src.schemas import Detection, FrameResult
from src.detector import YOLODetector, HelmetAssociator
from src.tracker import SortTracker, ViolationTracker
from src.utils.drawing import (
    draw_detection,
    draw_stats_overlay,
    COLOR_HAS_HELMET,
    COLOR_NO_HELMET,
    COLOR_HELMET_BOX,
    COLOR_MOTO_BOX,
)


class HelmetPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or default_settings

        self._detector   = YOLODetector(cfg)
        self._associator = HelmetAssociator(
            head_region_ratio=cfg.head_region_ratio,
            iou_threshold=cfg.helmet_iou_threshold,
        )
        self._tracker    = SortTracker(
            max_age=cfg.tracker_max_age,
            n_init=cfg.tracker_n_init,
        )
        self._violations = ViolationTracker(
            confirm_frames=cfg.violation_confirm_frames,
            output_dir=cfg.output_dir,
            save_snapshots=cfg.save_snapshots,
        )

        self._frame_count = 0
        self._fps_history: deque[float] = deque(maxlen=30)
        self._cfg = cfg

    # ------------------------------------------------------------------
    def process_image(self, frame: np.ndarray) -> FrameResult:
        all_dets   = self._detector.detect(frame)
        helmets    = [d for d in all_dets if d.class_id == self._cfg.helmet_class_id]
        no_helmets = [d for d in all_dets if d.class_id == self._cfg.no_helmet_class_id]

        violation_count = 0
        for det in helmets + no_helmets:
            has_helmet = (det.class_id == self._cfg.helmet_class_id)
            if not has_helmet:
                violation_count += 1
            color = COLOR_HAS_HELMET if has_helmet else COLOR_NO_HELMET
            label = f"CO MU ({det.confidence:.2f})" if has_helmet else f"KHONG MU ({det.confidence:.2f})"
            draw_detection(frame, det.bbox, label, color)

        draw_stats_overlay(frame, self._fps_history, 1,
                           violation_count, [])

        return FrameResult(
            frame_number=1,
            persons=helmets + no_helmets,
            helmets=helmets,
            new_violations=[],
            fps=0.0,
        )
    
    def process_frame(
        self,
        frame: np.ndarray,
        annotate: bool = True,
    ) -> FrameResult:
        t0 = time.perf_counter()
        self._frame_count += 1

        # 1. Detect
        all_dets = self._detector.detect(frame)

        # 2. Tách class
        helmets    = [d for d in all_dets if d.class_id == self._cfg.helmet_class_id]
        no_helmets = [d for d in all_dets if d.class_id == self._cfg.no_helmet_class_id]

        # 3. Track 
        tracked = self._tracker.update(helmets + no_helmets, frame)

        # 4. Evaluate violations & annotate
        new_violations: list[int] = []

        for det in tracked:
            has_helmet = (det.class_id == self._cfg.helmet_class_id)

            is_new = self._violations.update(
                track_id=det.track_id,
                has_helmet=has_helmet,
                frame_number=self._frame_count,
                bbox=det.bbox,
                confidence=det.confidence,
                frame=frame if annotate else None,
            )
            if is_new:
                new_violations.append(det.track_id)

            if annotate:
                if has_helmet:
                    draw_detection(frame, det.bbox, f"ID:{det.track_id} CO MU ({det.confidence:.2f})", COLOR_HAS_HELMET)
                else:
                    draw_detection(frame, det.bbox, f"ID:{det.track_id} KHONG MU ({det.confidence:.2f})", COLOR_NO_HELMET)
                    
        dt = time.perf_counter() - t0
        self._fps_history.append(1.0 / max(dt, 1e-6))

        if annotate:
            draw_stats_overlay(
                frame,
                self._fps_history,
                self._frame_count,
                self._violations.total_violations,
                new_violations,
            )

        return FrameResult(
            frame_number=self._frame_count,
            persons=tracked,
            helmets=helmets,
            new_violations=new_violations,
            fps=self._fps_history[-1] if self._fps_history else 0.0,
        )

    # ------------------------------------------------------------------
    @property
    def violation_tracker(self) -> ViolationTracker:
        return self._violations

    # ------------------------------------------------------------------
    @staticmethod
    def _find_person_index(
        tracked_det: Detection,
        original_persons: list[Detection],
    ) -> int:
        """Match a tracked detection back to its original person index."""
        from src.utils.geometry import compute_iou
        best_idx, best_iou = 0, 0.0
        for i, p in enumerate(original_persons):
            iou = compute_iou(tracked_det.bbox, p.bbox)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        return best_idx
