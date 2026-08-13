from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yolo11_deploy.detector import YOLODetector


class FakeModel:
    names = {0: "object"}

    def __init__(self, with_boxes: bool = True) -> None:
        self.with_boxes = with_boxes
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[object]:
        self.calls.append(kwargs)
        if not self.with_boxes:
            boxes = None
        else:
            boxes = FakeBoxes()
        return [SimpleNamespace(boxes=boxes, names=self.names)]


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = torch.tensor([[1.0, 2.0, 30.0, 40.0]])
        self.conf = torch.tensor([0.75])
        self.cls = torch.tensor([0.0])

    def __len__(self) -> int:
        return 1


def test_detector_returns_plain_detection_contract() -> None:
    model = FakeModel()
    detector = YOLODetector(model=model, confidence=0.3, iou=0.4, image_size=320)
    detections = detector.predict(np.zeros((64, 64, 3), dtype=np.uint8))
    assert detections == [
        {"class_id": 0, "class_name": "object", "confidence": pytest.approx(0.75), "bbox": [1, 2, 30, 40]}
    ]
    assert model.calls[0]["conf"] == 0.3
    assert model.calls[0]["imgsz"] == 320


def test_detector_validates_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        YOLODetector(model=FakeModel(), confidence=1.1)
