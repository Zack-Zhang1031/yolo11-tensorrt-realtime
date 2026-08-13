"""Reusable OpenCV stream processing, FPS smoothing, and writer configuration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from .protocols import FrameDetector
from .utils import ensure_parent
from .visualization import draw_detections

VIDEO_CODECS = {
    ".mp4": "mp4v",
    ".mov": "mp4v",
    ".avi": "MJPG",
}


@dataclass(slots=True)
class SmoothedFPS:
    """Exponential moving average of per-frame processing throughput."""

    alpha: float = 0.1
    value: float = 0.0

    def update(self, elapsed_seconds: float) -> float:
        current = 1.0 / max(elapsed_seconds, 1e-9)
        self.value = (
            current if self.value == 0.0 else (1 - self.alpha) * self.value + self.alpha * current
        )
        return self.value


def create_video_writer(
    output_path: str | Path,
    size: tuple[int, int],
    fps: float,
) -> tuple[cv2.VideoWriter, Path]:
    """Create an extension-appropriate video writer."""
    output = ensure_parent(output_path)
    codec = VIDEO_CODECS.get(output.suffix.lower())
    if codec is None:
        raise ValueError(f"Unsupported output extension: {output.suffix}")
    fourcc = cv2.VideoWriter_fourcc(*codec)  # pyright: ignore[reportAttributeAccessIssue]
    writer = cv2.VideoWriter(str(output), fourcc, fps, size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not create output video: {output}")
    return writer, output


def process_capture(
    capture: cv2.VideoCapture,
    detector: FrameDetector,
    *,
    window_name: str,
    writer: cv2.VideoWriter | None = None,
    display: bool = True,
    camera: bool = False,
) -> int:
    """Process frames and always release capture, writer, and OpenCV windows."""
    frames_written = 0
    fps = SmoothedFPS()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if camera:
                    print("Camera stopped returning frames; exiting.")
                break
            start = time.perf_counter()
            detections = detector.predict(frame)
            annotated = draw_detections(frame, detections, fps.update(time.perf_counter() - start))
            if writer is not None:
                writer.write(annotated)
                frames_written += 1
            if display:
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF in {27, ord("q"), ord("Q")}:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return frames_written
