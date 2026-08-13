"""TensorRT 10.x engine loading, CUDA memory management, and YOLO inference."""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .detector import Detection
from .postprocessing import decode_yolo_output
from .preprocessing import preprocess_image
from .utils import load_bgr_image, normalize_names


LOGGER = logging.getLogger(__name__)


def _load_cuda_runtime() -> Any:
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart
        except ImportError as exc:
            raise RuntimeError(
                "cuda-python is required for TensorRT inference. Install requirements-tensorrt.txt."
            ) from exc
    return cudart


def _check_cuda(result: tuple[Any, ...] | Any, operation: str) -> Any:
    """Check cuda-python's `(error, values...)` return convention."""
    values = result if isinstance(result, tuple) else (result,)
    error = values[0]
    if int(error) != 0:
        raise RuntimeError(f"CUDA operation {operation} failed with error code {int(error)}")
    if len(values) == 1:
        return None
    return values[1] if len(values) == 2 else values[1:]


class TensorRTDetector:
    """TensorRT 10.x inference runner for static-batch YOLO11 engines."""

    def __init__(
        self,
        engine_path: str | Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        class_names: dict[int, str] | list[str] | None = None,
    ) -> None:
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT is optional and requires NVIDIA GPU/CUDA/TensorRT."
            ) from exc
        major = int(trt.__version__.split(".", maxsplit=1)[0])
        if major < 10:
            raise RuntimeError(f"This runner targets TensorRT 10.x; found {trt.__version__}")

        path = Path(engine_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Non-empty TensorRT engine not found: {path}")
        self.trt = trt
        self.cudart = _load_cuda_runtime()
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")
        self.stream = _check_cuda(self.cudart.cudaStreamCreate(), "cudaStreamCreate")
        self.device_buffers: dict[str, int] = {}
        self.host_outputs: dict[str, np.ndarray] = {}
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [
            name
            for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [name for name in self.tensor_names if name not in self.input_names]
        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected one TensorRT input, found {self.input_names}")
        self.input_name = self.input_names[0]
        self.confidence = confidence
        self.iou = iou
        self.names = normalize_names(class_names or self._sidecar_names(path))
        self._closed = False
        self._allocate()

    @staticmethod
    def _sidecar_names(engine_path: Path) -> object:
        metadata_path = engine_path.with_suffix(".json")
        if not metadata_path.is_file():
            return {}
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            names = data.get("names", {})
            return ast.literal_eval(names) if isinstance(names, str) else names
        except (OSError, json.JSONDecodeError, SyntaxError, ValueError):
            LOGGER.warning("Could not parse TensorRT sidecar metadata: %s", metadata_path)
            return {}

    def _allocate(self) -> None:
        for name in self.tensor_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                if name == self.input_name:
                    profile_shape = self.engine.get_tensor_profile_shape(name, 0)[1]
                    self.context.set_input_shape(name, profile_shape)
                    shape = tuple(profile_shape)
                else:
                    shape = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"Unresolved dynamic TensorRT shape for {name}: {shape}")
            dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
            host = np.empty(shape, dtype=dtype)
            device = _check_cuda(self.cudart.cudaMalloc(host.nbytes), f"cudaMalloc({name})")
            self.device_buffers[name] = int(device)
            self.context.set_tensor_address(name, int(device))
            if name in self.output_names:
                self.host_outputs[name] = host
            elif name == self.input_name:
                self.input_shape = shape
                self.input_dtype = dtype

    def prepare(self, source: str | Path | np.ndarray) -> tuple[np.ndarray, object]:
        """Prepare a BGR image for this engine's NCHW input."""
        image = load_bgr_image(source)
        if len(self.input_shape) != 4 or self.input_shape[0] != 1 or self.input_shape[1] != 3:
            raise ValueError(f"Expected static [1,3,H,W] TensorRT input, got {self.input_shape}")
        return preprocess_image(image, self.input_shape[-2:], self.input_dtype)

    def infer_raw(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Copy input, enqueue inference, copy outputs, and synchronize the CUDA stream."""
        tensor = np.ascontiguousarray(tensor, dtype=self.input_dtype)
        expected = int(np.prod(self.input_shape))
        if tensor.size != expected:
            raise ValueError(f"Input has {tensor.size} values; engine expects {expected}")
        kind = self.cudart.cudaMemcpyKind
        _check_cuda(
            self.cudart.cudaMemcpyAsync(
                self.device_buffers[self.input_name],
                tensor.ctypes.data,
                tensor.nbytes,
                kind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "input cudaMemcpyAsync",
        )
        if not self.context.execute_async_v3(stream_handle=self.stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        for name, host in self.host_outputs.items():
            _check_cuda(
                self.cudart.cudaMemcpyAsync(
                    host.ctypes.data,
                    self.device_buffers[name],
                    host.nbytes,
                    kind.cudaMemcpyDeviceToHost,
                    self.stream,
                ),
                f"output cudaMemcpyAsync({name})",
            )
        self.synchronize()
        return [self.host_outputs[name].copy() for name in self.output_names]

    def synchronize(self) -> None:
        """Synchronize the runner's CUDA stream for correct timing."""
        _check_cuda(self.cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")

    def predict(self, source: str | Path | np.ndarray) -> list[Detection]:
        """Run the complete TensorRT YOLO image pipeline."""
        tensor, info = self.prepare(source)
        outputs = self.infer_raw(tensor)
        if not self.names:
            channels = min(np.asarray(outputs[0]).shape[-2:])
            self.names = normalize_names(None, max(0, channels - 4))
        return decode_yolo_output(outputs, info, self.names, self.confidence, self.iou)

    def close(self) -> None:
        """Release CUDA allocations and the stream."""
        if self._closed:
            return
        for device in self.device_buffers.values():
            _check_cuda(self.cudart.cudaFree(device), "cudaFree")
        _check_cuda(self.cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
        self.device_buffers.clear()
        self._closed = True

    def __enter__(self) -> "TensorRTDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.close()
            except Exception:
                pass

