"""Run TensorRT 10.x YOLO11 inference and save an annotated image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from yolo11_deploy.tensorrt_runtime import TensorRTDetector
from yolo11_deploy.utils import configure_logging, ensure_parent, load_bgr_image
from yolo11_deploy.visualization import draw_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/tensorrt-detection.jpg"))
    parser.add_argument("--imgsz", type=int, help="Runtime size for a dynamic engine")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    image = load_bgr_image(args.source)
    with TensorRTDetector(args.engine, args.conf, args.iou, image_size=args.imgsz) as detector:
        detections = detector.predict(image)
    output = ensure_parent(args.output)
    if not cv2.imwrite(str(output), draw_detections(image, detections)):
        raise RuntimeError(f"OpenCV failed to write {output}")
    print(json.dumps(detections, indent=2, ensure_ascii=False))
    print(f"Saved annotated image: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
