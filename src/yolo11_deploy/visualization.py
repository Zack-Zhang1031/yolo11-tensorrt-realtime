"""OpenCV visualization helpers."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    fps: float | None = None,
) -> np.ndarray:
    """Draw detection boxes, labels, confidence values, and optional FPS."""
    canvas = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection["bbox"])
        color = _class_color(detection["class_id"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f'{detection["class_name"]} {detection["confidence"]:.2f}'
        (width, height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(0, y1 - height - baseline - 6)
        cv2.rectangle(canvas, (x1, label_top), (x1 + width + 6, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, max(height + 1, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if fps is not None:
        cv2.putText(
            canvas,
            f"FPS: {fps:.1f}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _class_color(class_id: int) -> tuple[int, int, int]:
    return ((37 * class_id + 80) % 255, (17 * class_id + 160) % 255, (29 * class_id + 220) % 255)

