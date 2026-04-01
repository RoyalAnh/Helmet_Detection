#!/usr/bin/env python
"""
Run the helmet detection pipeline from the command line.

Examples
--------
  # webcam
  python scripts/run.py

  # video file, save annotated output
  python scripts/run.py --source traffic.mp4 --output output/result.mp4

  # image
  python scripts/run.py --source photo.jpg
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from loguru import logger

from configs.settings import settings
from src.pipeline import HelmetPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",
                   help="Video file path, image path, or camera index (default: 0)")
    p.add_argument("--output", default=None,
                   help="Save annotated video to this path (optional)")
    p.add_argument("--no-display", action="store_true",
                   help="Suppress OpenCV window (useful on headless servers)")
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    pipeline = HelmetPipeline(settings)

    # ── Image mode ──────────────────────────────────────────────────────────
    if isinstance(source, str) and Path(source).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }:
        frame = cv2.imread(source)
        if frame is None:
            logger.error(f"Cannot read image: {source}")
            sys.exit(1)

        pipeline.process_image(frame)
        if not args.no_display:
            cv2.imshow("Helmet Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ── Video / camera mode ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_in, (w, h))

    logger.info(f"Source: {source}  {w}x{h} @ {fps_in:.0f}fps")
    logger.info("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = pipeline.process_frame(frame, annotate=True)
            if result.new_violations:
                logger.warning(f"New violations! Track IDs: {result.new_violations}")
            if writer:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Helmet Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        log_path = pipeline.violation_tracker.export_log()
        logger.info(f"Violation log saved → {log_path}")
        logger.info(f"Total violations: {pipeline.violation_tracker.total_violations}")


if __name__ == "__main__":
    main()
