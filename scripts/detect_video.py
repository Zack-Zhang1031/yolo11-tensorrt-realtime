"""Run YOLO11 detection on a video with optional output recording."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from yolo11_deploy.detector import YOLODetector
from yolo11_deploy.utils import configure_logging, ensure_parent
from yolo11_deploy.visualization import draw_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save", nargs="?", const="outputs/detected-video.mp4")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    if args.source.suffix.lower() not in {".mp4", ".avi", ".mov"}:
        raise ValueError("Supported video extensions: .mp4, .avi, .mov")
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        print(f"Could not open video: {args.source.resolve()}")
        return 2
    writer = None
    try:
        detector = YOLODetector(args.model, args.device, args.conf, args.iou, args.imgsz)
        if args.save:
            output = ensure_parent(args.save)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {output}")
        smoothed_fps = 0.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            start = time.perf_counter()
            detections = detector.predict(frame)
            current_fps = 1.0 / max(time.perf_counter() - start, 1e-9)
            smoothed_fps = (
                current_fps
                if smoothed_fps == 0
                else 0.9 * smoothed_fps + 0.1 * current_fps
            )
            annotated = draw_detections(frame, detections, smoothed_fps)
            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("YOLO11 Video Detection", annotated)
                if cv2.waitKey(1) & 0xFF in {27, ord("q"), ord("Q")}:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    if args.save:
        print(f"Saved detected video: {Path(args.save).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
