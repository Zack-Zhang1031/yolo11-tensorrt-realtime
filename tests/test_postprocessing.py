import numpy as np
import pytest

from yolo11_deploy.postprocessing import decode_yolo_output
from yolo11_deploy.preprocessing import LetterboxInfo


def test_decode_transposed_yolo_output_and_class_aware_nms() -> None:
    output = np.zeros((1, 6, 3), dtype=np.float32)
    output[0, :, 0] = [100, 100, 40, 40, 0.9, 0.1]
    output[0, :, 1] = [102, 102, 40, 40, 0.8, 0.2]
    output[0, :, 2] = [300, 300, 20, 20, 0.1, 0.95]
    info = LetterboxInfo((640, 640), (640, 640), 1.0, 0.0, 0.0)
    detections = decode_yolo_output(output, info, {0: "a", 1: "b"}, 0.25, 0.5)
    assert len(detections) == 2
    assert [item["class_name"] for item in detections] == ["b", "a"]
    assert detections[1]["bbox"] == pytest.approx([80, 80, 120, 120])


def test_decode_maps_letterbox_coordinates_back() -> None:
    output = np.array([[[320], [320], [320], [160], [0.9]]], dtype=np.float32)
    info = LetterboxInfo((100, 200), (640, 640), 3.2, 0.0, 160.0)
    detection = decode_yolo_output(output, info, {0: "object"})[0]
    assert detection["bbox"] == pytest.approx([50, 25, 150, 75])


def test_decode_empty_confidence_returns_empty_list() -> None:
    output = np.zeros((1, 5, 10), dtype=np.float32)
    info = LetterboxInfo((640, 640), (640, 640), 1.0, 0.0, 0.0)
    assert decode_yolo_output(output, info, {0: "object"}, confidence=0.5) == []

