"""
Associates detected helmets with persons using IoU on the head region.

Strategy
--------
1. Compute head_region = top `head_ratio` fraction of each person bbox.
2. For every person, check whether any helmet bbox overlaps its head_region
   with IoU >= threshold.
3. Returns a bool mask: person_index -> has_helmet.
"""
from __future__ import annotations

from src.schemas import Detection
from src.utils.geometry import compute_iou


class HelmetAssociator:
    def __init__(self, head_region_ratio: float, iou_threshold: float) -> None:
        self._head_ratio = head_region_ratio
        self._iou_threshold = iou_threshold

    # ------------------------------------------------------------------
    def associate(
        self,
        persons: list[Detection],
        helmets: list[Detection],
    ) -> dict[int, bool]:
        """Return {person_index: has_helmet}."""
        result: dict[int, bool] = {}
        for i, person in enumerate(persons):
            head = self._head_region(person.bbox)
            result[i] = any(
                compute_iou(head, h.bbox) >= self._iou_threshold
                for h in helmets
            )
        return result

    # ------------------------------------------------------------------
    def _head_region(self, bbox: list[float]) -> list[float]:
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2, y1 + (y2 - y1) * self._head_ratio]
