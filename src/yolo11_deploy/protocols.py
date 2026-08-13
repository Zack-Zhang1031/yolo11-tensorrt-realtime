"""Structural interfaces used to decouple adapters from third-party implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .detector import Detection


class UltralyticsModel(Protocol):
    """Minimal Ultralytics model surface required by :class:`YOLODetector`."""

    names: object

    def predict(self, **kwargs: object) -> list[Any]: ...


class FrameDetector(Protocol):
    """Backend-independent detector accepted by video processing utilities."""

    def predict(self, source: str | Path | np.ndarray) -> list[Detection]: ...
