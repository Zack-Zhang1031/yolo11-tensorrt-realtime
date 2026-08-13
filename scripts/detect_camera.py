"""Run real-time YOLO11 detection from an OpenCV camera."""

from __future__ import annotations

import argparse

import cv2

from yolo11_deploy.detector import YOLODetector
from yolo11_deploy.utils import configure_logging
from yolo11_deploy.video import process_capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        print(f"Could not open camera index {args.camera}. Check camera permissions and index.")
        return 2
    try:
        detector = YOLODetector(args.model, args.device, args.conf, args.iou, args.imgsz)
        process_capture(
            capture,
            detector,
            window_name="YOLO11 Camera Detection",
            camera=True,
        )
    except Exception:
        if capture.isOpened():
            capture.release()
        cv2.destroyAllWindows()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
