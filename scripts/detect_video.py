"""Run YOLO11 detection on a video with optional output recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from yolo11_deploy.detector import YOLODetector
from yolo11_deploy.utils import configure_logging
from yolo11_deploy.video import VIDEO_CODECS, create_video_writer, process_capture


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
    if args.source.suffix.lower() not in VIDEO_CODECS:
        raise ValueError(f"Supported video extensions: {', '.join(VIDEO_CODECS)}")
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        capture.release()
        print(f"Could not open video: {args.source.resolve()}")
        return 2

    writer = None
    output: Path | None = None
    try:
        detector = YOLODetector(args.model, args.device, args.conf, args.iou, args.imgsz)
        if args.save:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            writer, output = create_video_writer(
                args.save,
                (width, height),
                source_fps if source_fps > 0 else 30.0,
            )
        frames_written = process_capture(
            capture,
            detector,
            window_name="YOLO11 Video Detection",
            writer=writer,
            display=not args.no_display,
        )
    except Exception:
        if capture.isOpened():
            capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        raise

    if output is not None:
        if frames_written == 0:
            raise RuntimeError(f"No frames were written to output video: {output}")
        print(f"Saved {frames_written} detected frames: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
