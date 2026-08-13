"""ONNX Runtime inference adapter for raw Ultralytics detection exports."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import numpy as np

from .detector import Detection
from .postprocessing import decode_yolo_output
from .preprocessing import LetterboxInfo, preprocess_image
from .utils import load_bgr_image, normalize_names

LOGGER = logging.getLogger(__name__)


class ONNXDetector:
    """Run a YOLO11 ONNX model using CPU or CUDA Execution Provider."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = "auto",
        class_names: dict[int, str] | list[str] | None = None,
        image_size: int | tuple[int, int] = 640,
        device_id: int = 0,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime is not installed. Install requirements.txt.") from exc

        path = Path(model_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Non-empty ONNX model not found: {path}")
        available = ort.get_available_providers()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if device == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider was requested but is not available")
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        if device in {"auto", "cuda"} and "CUDAExecutionProvider" in available:
            preload = getattr(ort, "preload_dlls", None)
            if callable(preload):
                preload()
            cuda_provider = ("CUDAExecutionProvider", {"device_id": device_id})
            providers = (
                [cuda_provider] if device == "cuda" else [cuda_provider, "CPUExecutionProvider"]
            )
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=providers)
        if device == "cuda":
            active = self.session.get_providers()
            if not active or active[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"CUDAExecutionProvider was not activated: {active}")
            disable_fallback = getattr(self.session, "disable_fallback", None)
            if callable(disable_fallback):
                disable_fallback()
        self.input = self.session.get_inputs()[0]
        if self.input.type not in {"tensor(float)", "tensor(float16)"}:
            raise ValueError(f"Unsupported ONNX input type: {self.input.type}")
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.confidence = confidence
        self.iou = iou
        self.providers = self.session.get_providers()
        self.device_id = device_id if self.providers[0] == "CUDAExecutionProvider" else None
        self.names = normalize_names(class_names or self._metadata_names())
        self.input_size = self._input_size(image_size)
        LOGGER.info("ONNX Runtime providers: %s", self.providers)

    def _metadata_names(self) -> object:
        raw = self.session.get_modelmeta().custom_metadata_map.get("names", "")
        if not raw:
            return {}
        try:
            return ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            LOGGER.warning("Could not parse class names from ONNX metadata")
            return {}

    def _input_size(self, fallback: int | tuple[int, int]) -> tuple[int, int]:
        shape = self.input.shape
        if len(shape) != 4:
            raise ValueError(f"Expected NCHW ONNX input, got {shape}")
        height, width = shape[-2:]
        fallback_height, fallback_width = (
            (fallback, fallback) if isinstance(fallback, int) else fallback
        )
        resolved_height = height if isinstance(height, int) and height > 0 else fallback_height
        resolved_width = width if isinstance(width, int) and width > 0 else fallback_width
        if resolved_height <= 0 or resolved_width <= 0:
            raise ValueError("ONNX inference dimensions must be positive")
        return resolved_height, resolved_width

    def prepare(self, source: str | Path | np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        """Prepare one source image for inference."""
        image = load_bgr_image(source)
        dtype = np.float16 if self.input.type == "tensor(float16)" else np.float32
        return preprocess_image(image, self.input_size, np.dtype(dtype))

    def infer_raw(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Run ONNX Runtime without postprocessing, for benchmarks and diagnostics."""
        return self.session.run(self.output_names, {self.input.name: tensor})

    def predict(self, source: str | Path | np.ndarray) -> list[Detection]:
        """Run preprocessing, ONNX inference, NMS, and source-coordinate projection."""
        tensor, info = self.prepare(source)
        outputs = self.infer_raw(tensor)
        if not self.names:
            output = np.asarray(outputs[0])
            channels = min(output.shape[-2:])
            self.names = normalize_names(None, max(0, channels - 4))
        return decode_yolo_output(outputs, info, self.names, self.confidence, self.iou)
