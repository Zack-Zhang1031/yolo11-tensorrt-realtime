"""Clean Ultralytics YOLO adapter used by image, video, and camera applications."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from .utils import load_bgr_image, normalize_names


class Detection(TypedDict):
    """Backend-independent object detection result."""

    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class YOLODetector:
    """Adapt Ultralytics YOLO predictions to plain Python dictionaries."""

    def __init__(
        self,
        model_path: str | Path = "yolo11s.pt",
        device: str = "cpu",
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        half: bool = False,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0 or not 0.0 <= iou <= 1.0:
            raise ValueError("Confidence and IoU thresholds must be within [0, 1]")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.model_path = Path(model_path)
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.half = half

        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is required for PyTorch inference. Install requirements.txt."
                ) from exc
            try:
                model = YOLO(str(model_path))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load {model_path}. If this is the first run, verify network access "
                    "so Ultralytics can download official weights."
                ) from exc
        self.model = model
        self.names = normalize_names(getattr(model, "names", {}))

    def predict(self, source: str | Path | np.ndarray) -> list[Detection]:
        """Run one-image inference and return backend-independent detections."""
        image = load_bgr_image(source)
        predict_options: dict[str, object] = dict(
            source=image,
            device=self.device,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            verbose=False,
        )
        if self.half:
            predict_options["half"] = True
        results = self.model.predict(**predict_options)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        names = normalize_names(getattr(result, "names", self.names))
        return [
            {
                "class_id": int(class_id),
                "class_name": names.get(int(class_id), f"class_{int(class_id)}"),
                "confidence": float(score),
                "bbox": [float(value) for value in box],
            }
            for box, score, class_id in zip(xyxy, confidences, class_ids, strict=True)
        ]
