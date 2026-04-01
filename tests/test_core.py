"""
Unit tests — run with:  pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.utils.geometry import compute_iou, get_head_region
from src.detector.helmet_associator import HelmetAssociator
from src.tracker.violation_tracker import ViolationTracker
from src.schemas import Detection


# ── Geometry ─────────────────────────────────────────────────────────────────

class TestComputeIou:
    def test_perfect_overlap(self):
        b = [0, 0, 10, 10]
        assert compute_iou(b, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_iou([0, 0, 5, 5], [10, 10, 20, 20]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.1 < iou < 0.2   # 25/175 ≈ 0.143

    def test_contained_box(self):
        outer = [0, 0, 10, 10]
        inner = [2, 2, 8, 8]
        iou = compute_iou(outer, inner)
        assert iou == pytest.approx(36 / 100)


class TestGetHeadRegion:
    def test_top_30_percent(self):
        head = get_head_region([0, 0, 100, 200], ratio=0.30)
        assert head == [0, 0, 100, 60]

    def test_preserves_x(self):
        head = get_head_region([10, 20, 80, 120])
        assert head[0] == 10 and head[2] == 80


# ── Helmet Associator ────────────────────────────────────────────────────────

class TestHelmetAssociator:
    def _make(self, head_ratio=0.30, iou_thresh=0.15):
        return HelmetAssociator(head_ratio, iou_thresh)

    def _person(self, bbox):
        return Detection(bbox, class_id=0, class_name="person", confidence=0.9)

    def _helmet(self, bbox):
        return Detection(bbox, class_id=1, class_name="helmet", confidence=0.8)

    def test_helmet_on_head(self):
        assoc = self._make()
        # Person bbox: full body; helmet overlaps the top region
        person = self._person([100, 100, 200, 400])
        helmet = self._helmet([105, 102, 195, 175])  # clearly in head region
        result = assoc.associate([person], [helmet])
        assert result[0] is True

    def test_no_helmet(self):
        assoc = self._make()
        person = self._person([100, 100, 200, 400])
        helmet = self._helmet([300, 300, 400, 380])  # far away
        result = assoc.associate([person], [helmet])
        assert result[0] is False

    def test_empty_inputs(self):
        assoc = self._make()
        assert assoc.associate([], []) == {}


# ── Violation Tracker ────────────────────────────────────────────────────────

class TestViolationTracker:
    def _make(self, confirm=3, output_dir="/tmp/test_violations"):
        return ViolationTracker(confirm, output_dir, save_snapshots=False)

    def test_not_confirmed_before_n_frames(self):
        vt = self._make(confirm=3)
        for _ in range(2):
            is_new = vt.update(1, False, 1, [0, 0, 10, 10], 0.9)
            assert is_new is False

    def test_confirmed_after_n_frames(self):
        vt = self._make(confirm=3)
        results = [vt.update(1, False, i, [0, 0, 10, 10], 0.9) for i in range(3)]
        assert results[-1] is True

    def test_no_duplicate_confirmation(self):
        vt = self._make(confirm=2)
        vt.update(1, False, 1, [0, 0, 10, 10], 0.9)
        first = vt.update(1, False, 2, [0, 0, 10, 10], 0.9)
        second = vt.update(1, False, 3, [0, 0, 10, 10], 0.9)
        assert first is True
        assert second is False  # already confirmed — don't re-fire

    def test_reset_by_helmet_frame(self):
        vt = self._make(confirm=3)
        vt.update(1, False, 1, [0, 0, 10, 10], 0.9)
        vt.update(1, False, 2, [0, 0, 10, 10], 0.9)
        vt.update(1, True,  3, [0, 0, 10, 10], 0.9)   # helmet seen — breaks streak
        result = vt.update(1, False, 4, [0, 0, 10, 10], 0.9)
        assert result is False   # streak reset; only 1 consecutive no-helmet frame
