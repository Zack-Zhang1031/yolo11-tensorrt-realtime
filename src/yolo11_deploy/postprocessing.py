"""Shape-aware YOLO detection output decoding and non-maximum suppression."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .detector import Detection
from .preprocessing import LetterboxInfo
from .utils import normalize_names


def _prediction_rows(output: np.ndarray, class_count: int | None) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported, got {array.shape}")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D YOLO output, got {array.shape}")

    expected_channels = 4 + class_count if class_count else None
    if expected_channels and array.shape[0] == expected_channels:
        return array.T
    if expected_channels and array.shape[1] == expected_channels:
        return array
    # YOLO export is commonly [channels, anchors]; choose the smaller axis as channels.
    return array.T if array.shape[0] < array.shape[1] else array


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes, dtype=np.float32)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def _scale_boxes(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    boxes = boxes.astype(np.float32, copy=True)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.pad_x) / info.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.pad_y) / info.scale
    height, width = info.original_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes


def decode_yolo_output(
    outputs: np.ndarray | Sequence[np.ndarray],
    info: LetterboxInfo,
    class_names: dict[int, str] | list[str] | tuple[str, ...],
    confidence: float = 0.25,
    iou: float = 0.45,
) -> list[Detection]:
    """Decode a raw Ultralytics detection export, apply class-aware NMS, and rescale."""
    if not 0.0 <= confidence <= 1.0 or not 0.0 <= iou <= 1.0:
        raise ValueError("Confidence and IoU thresholds must be within [0, 1]")
    output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    names = normalize_names(class_names)
    rows = _prediction_rows(np.asarray(output), len(names) or None)
    if rows.shape[1] < 5:
        raise ValueError(f"Detection output requires at least 5 channels, got {rows.shape}")

    scores = rows[:, 4:]
    if names and scores.shape[1] != len(names):
        raise ValueError(
            f"Output has {scores.shape[1]} class channels but {len(names)} names were supplied"
        )
    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]
    keep = confidences >= confidence
    if not np.any(keep):
        return []

    boxes = _xywh_to_xyxy(rows[keep, :4])
    class_ids = class_ids[keep]
    confidences = confidences[keep]
    selected: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        xywh = boxes[indices].copy()
        xywh[:, 2] -= xywh[:, 0]
        xywh[:, 3] -= xywh[:, 1]
        kept = cv2.dnn.NMSBoxes(xywh.tolist(), confidences[indices].tolist(), confidence, iou)
        if len(kept):
            selected.extend(indices[np.asarray(kept).reshape(-1)].tolist())

    selected = sorted(selected, key=lambda index: float(confidences[index]), reverse=True)
    boxes = _scale_boxes(boxes, info)
    return [
        {
            "class_id": int(class_ids[index]),
            "class_name": names.get(int(class_ids[index]), f"class_{int(class_ids[index])}"),
            "confidence": float(confidences[index]),
            "bbox": [float(value) for value in boxes[index]],
        }
        for index in selected
    ]
