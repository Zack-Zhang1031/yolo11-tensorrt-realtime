import numpy as np
import pytest

from yolo11_deploy.preprocessing import letterbox, preprocess_image


def test_letterbox_preserves_aspect_ratio_and_records_padding() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    output, info = letterbox(image, 640)
    assert output.shape == (640, 640, 3)
    assert info.scale == pytest.approx(3.2)
    assert info.pad_x == 0
    assert info.pad_y == 160


def test_preprocess_produces_rgb_nchw_float() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    tensor, _ = preprocess_image(image, (32, 32))
    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0 and tensor.max() <= 1
    assert tensor[0, 0, 16, 16] == pytest.approx(1.0)


def test_letterbox_rejects_invalid_image() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        letterbox(np.zeros((10, 10), dtype=np.uint8))

