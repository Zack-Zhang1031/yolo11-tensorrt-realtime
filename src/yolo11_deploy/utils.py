"""Shared path, logging, dependency, and image helpers."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LOGGER = logging.getLogger("yolo11_deploy")


def configure_logging(verbose: bool = False) -> None:
    """Configure concise console logging for command-line tools."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def optional_dependency_available(module_name: str) -> bool:
    """Return whether an optional import can be resolved without importing it."""
    return importlib.util.find_spec(module_name) is not None


def ensure_parent(path: str | Path) -> Path:
    """Create a file's parent directory and return an absolute path."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_bgr_image(source: str | Path | np.ndarray) -> np.ndarray:
    """Load a BGR image from a path or validate an existing NumPy image."""
    if isinstance(source, np.ndarray):
        image = source
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path.resolve()}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"OpenCV could not decode image: {path.resolve()}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 BGR image, got shape {image.shape}")
    return np.ascontiguousarray(image)


def normalize_names(names: Any, class_count: int | None = None) -> dict[int, str]:
    """Normalize model metadata class names into an integer-keyed mapping."""
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    count = class_count or 0
    return {index: f"class_{index}" for index in range(count)}

