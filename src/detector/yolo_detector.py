"""
Thin wrapper around ultralytics YOLOv8.
Responsible only for inference; returns typed Detection objects.
"""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from configs.settings import Settings
from src.schemas import Detection


class YOLODetector:
    def __init__(self, settings: Settings) -> None:
        self._cfg = settings
        self._model = YOLO(settings.model_path)
        self._model.to(settings.device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on one BGR frame; return all detections."""
        results = self._model.predict(
            frame,
            conf=self._cfg.confidence_threshold,
            iou=self._cfg.iou_threshold,
            imgsz=self._cfg.input_size,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            name = results.names.get(cls_id, str(cls_id))
            detections.append(Detection([x1, y1, x2, y2], cls_id, name, conf))

        return detections

