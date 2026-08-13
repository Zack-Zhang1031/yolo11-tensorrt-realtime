"""Run real-time YOLO11 detection from an OpenCV camera."""

from __future__ import annotations

import argparse
import time

import cv2

from yolo11_deploy.detector import YOLODetector
from yolo11_deploy.utils import configure_logging
from yolo11_deploy.visualization import draw_detections


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
    detector = YOLODetector(args.model, args.device, args.conf, args.iou, args.imgsz)
    smoothed_fps = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera stopped returning frames; exiting.")
                break
            start = time.perf_counter()
            detections = detector.predict(frame)
            current_fps = 1.0 / max(time.perf_counter() - start, 1e-9)
            smoothed_fps = current_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * current_fps
            cv2.imshow("YOLO11 Camera Detection", draw_detections(frame, detections, smoothed_fps))
            if cv2.waitKey(1) & 0xFF in {27, ord("q"), ord("Q")}:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

