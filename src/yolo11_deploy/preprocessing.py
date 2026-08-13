"""Image preprocessing shared by ONNX Runtime and TensorRT inference."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class LetterboxInfo:
    """Geometry required to project predictions back to the source image."""

    original_shape: tuple[int, int]
    input_shape: tuple[int, int]
    scale: float
    pad_x: float
    pad_y: float


def letterbox(
    image: np.ndarray,
    size: int | tuple[int, int] = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize with unchanged aspect ratio and symmetric padding."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {image.shape}")
    input_h, input_w = (size, size) if isinstance(size, int) else size
    if input_h <= 0 or input_w <= 0:
        raise ValueError("Input size must be positive")

    original_h, original_w = image.shape[:2]
    scale = min(input_w / original_w, input_h / original_h)
    resized_w = max(1, round(original_w * scale))
    resized_h = max(1, round(original_h * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = input_w - resized_w
    pad_h = input_h - resized_h
    left = pad_w // 2
    top = pad_h // 2
    output = cv2.copyMakeBorder(
        resized,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    info = LetterboxInfo(
        original_shape=(original_h, original_w),
        input_shape=(input_h, input_w),
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
    )
    return output, info


def preprocess_image(
    image: np.ndarray,
    size: int | tuple[int, int] = 640,
    dtype: np.dtype | None = None,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Convert BGR HWC uint8 input to normalized RGB NCHW tensor."""
    dtype = dtype or np.dtype(np.float32)
    padded, info = letterbox(image, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1)[None]
    tensor = np.ascontiguousarray(tensor, dtype=dtype) / np.array(255.0, dtype=dtype)
    return tensor, info
