"""OpenCV drawing utilities for annotating frames."""
from __future__ import annotations

import cv2
import numpy as np
from collections import deque

# Color palette  (BGR)
COLOR_HAS_HELMET = (34, 197, 94)     # green
COLOR_NO_HELMET  = (39,  39, 225)    # red
COLOR_HELMET_BOX = (0,  200, 255)    # yellow
COLOR_MOTO_BOX   = (60, 120, 200)    # brown-ish


def draw_detection(
    frame: np.ndarray,
    bbox: list[float],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw bbox + label in-place (modifies frame)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font_scale = 0.4
    font_thickness = 1
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    # label position with y (bounding box)
    if y1 - th - 4 >= 0:
        y_label = y1 - th - 4  # trên
        y_text = y_label + th
    else:
        y_label = y2 + 4       # dưới
        y_text = y_label + th

    # label position with x (avoid overtaking on the right (image))
    x_label = x1
    if x1 + tw + 4 > w:
        x_label = max(0, w - tw - 4)  # move inward

    # Draw rectangle + text
    cv2.rectangle(frame, (x_label, y_label), (x_label + tw + 4, y_label + th + 4), color, -1)
    cv2.putText(frame, label, (x_label + 2, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
def draw_stats_overlay(
    frame: np.ndarray,
    fps_history: deque,
    frame_number: int,
    total_violations: int,
    new_violations: list[int],
) -> None:
    """Draw a semi-transparent stats panel in the top-left corner (in-place)."""
    avg_fps = float(np.mean(fps_history)) if fps_history else 0.0
    lines = [
        f"FPS: {avg_fps:.1f}",
        f"Frame: {frame_number}",
        f"Violations: {total_violations}",
    ]
    if new_violations:
        lines.append(f"NEW! IDs: {new_violations}")

    panel_h = 22 + len(lines) * 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (240, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        color = (60, 80, 255) if "NEW!" in line else (210, 210, 210)
        cv2.putText(
            frame, line, (8, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
        )
