"""ONNX Runtime inference adapter for raw Ultralytics detection exports."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import numpy as np

from .detector import Detection
from .postprocessing import decode_yolo_output
from .preprocessing import preprocess_image
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
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device in {"auto", "cuda"} and "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input = self.session.get_inputs()[0]
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.confidence = confidence
        self.iou = iou
        self.providers = self.session.get_providers()
        self.names = normalize_names(class_names or self._metadata_names())
        self.input_size = self._input_size()
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

    def _input_size(self) -> tuple[int, int]:
        shape = self.input.shape
        if len(shape) != 4:
            raise ValueError(f"Expected NCHW ONNX input, got {shape}")
        height, width = shape[-2:]
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError("Dynamic ONNX image dimensions require an explicit profile and are unsupported")
        return height, width

    def prepare(self, source: str | Path | np.ndarray) -> tuple[np.ndarray, object]:
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

