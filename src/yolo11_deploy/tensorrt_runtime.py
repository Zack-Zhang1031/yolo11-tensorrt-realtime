"""TensorRT 10.x engine loading, CUDA memory management, and YOLO inference."""

from __future__ import annotations

import ast
import ctypes
import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .detector import Detection
from .postprocessing import decode_yolo_output
from .preprocessing import LetterboxInfo, preprocess_image
from .utils import load_bgr_image, normalize_names

LOGGER = logging.getLogger(__name__)


def _load_cuda_runtime() -> Any:
    try:
        return importlib.import_module("cuda.bindings.runtime")
    except ImportError:
        try:
            return importlib.import_module("cuda.cudart")
        except ImportError as exc:
            raise RuntimeError(
                "cuda-python is required for TensorRT inference. Install requirements-tensorrt.txt."
            ) from exc


def _check_cuda(result: tuple[Any, ...] | Any, operation: str) -> Any:
    """Check cuda-python's `(error, values...)` return convention."""
    values = result if isinstance(result, tuple) else (result,)
    error = values[0]
    if int(error) != 0:
        raise RuntimeError(f"CUDA operation {operation} failed with error code {int(error)}")
    if len(values) == 1:
        return None
    return values[1] if len(values) == 2 else values[1:]


def require_cuda_device() -> tuple[Any, int]:
    """Load the CUDA runtime and require at least one visible CUDA device."""
    cudart = _load_cuda_runtime()
    device_count = int(_check_cuda(cudart.cudaGetDeviceCount(), "cudaGetDeviceCount"))
    if device_count < 1:
        raise RuntimeError("TensorRT requires at least one visible NVIDIA CUDA device.")
    return cudart, device_count


@dataclass(slots=True)
class TensorBuffer:
    """TensorRT I/O tensor allocation with pinned host and CUDA device memory."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    host: np.ndarray
    device: int


class TensorRTDetector:
    """TensorRT 10.x inference runner for trusted YOLO11 engine files."""

    def __init__(
        self,
        engine_path: str | Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        class_names: dict[int, str] | list[str] | None = None,
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        self._closed = False
        self.cudart: Any = None
        self.logger: Any = None
        self.runtime: Any = None
        self.engine: Any = None
        self.context: Any = None
        self.stream: Any = None
        self.buffers: dict[str, TensorBuffer] = {}
        try:
            self._initialize(engine_path, confidence, iou, class_names, image_size)
        except Exception:
            self.close(suppress_errors=True)
            raise

    def _initialize(
        self,
        engine_path: str | Path,
        confidence: float,
        iou: float,
        class_names: dict[int, str] | list[str] | None,
        image_size: int | tuple[int, int] | None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0 or not 0.0 <= iou <= 1.0:
            raise ValueError("Confidence and IoU thresholds must be within [0, 1]")
        try:
            import tensorrt as trt  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError("TensorRT 10.x and an NVIDIA CUDA environment are required") from exc
        major = int(trt.__version__.split(".", maxsplit=1)[0])
        if major != 10:
            raise RuntimeError(f"This runner supports TensorRT 10.x; found {trt.__version__}")

        path = Path(engine_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Non-empty TensorRT engine not found: {path}")
        LOGGER.warning("Deserialize only TensorRT engines from a trusted source: %s", path)
        self.trt = trt
        self.cudart, _ = require_cuda_device()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")
        self.stream = int(_check_cuda(self.cudart.cudaStreamCreate(), "cudaStreamCreate"))
        self.tensor_names = [
            self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)
        ]
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
        self.set_input_shape(image_size)

    @staticmethod
    def _sidecar_names(engine_path: Path) -> object:
        metadata_path = engine_path.with_suffix(f"{engine_path.suffix}.json")
        if not metadata_path.is_file():
            return {}
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            names = data.get("names", {})
            return ast.literal_eval(names) if isinstance(names, str) else names
        except (OSError, json.JSONDecodeError, SyntaxError, ValueError):
            LOGGER.warning("Could not parse TensorRT sidecar metadata: %s", metadata_path)
            return {}

    def set_input_shape(self, size: int | tuple[int, int] | None = None) -> None:
        """Select a static or profile-valid dynamic shape and reallocate all I/O buffers."""
        self._require_open()
        engine_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        is_dynamic = any(dimension < 0 for dimension in engine_shape)
        if is_dynamic:
            if size is None:
                target_shape = tuple(self.engine.get_tensor_profile_shape(self.input_name, 0)[1])
            else:
                height, width = (size, size) if isinstance(size, int) else size
                target_shape = (1, 3, height, width)
            minimum, _, maximum = self.engine.get_tensor_profile_shape(self.input_name, 0)
            if any(
                value < low or value > high
                for value, low, high in zip(target_shape, minimum, maximum, strict=True)
            ):
                raise ValueError(
                    f"Input shape {target_shape} is outside profile range {minimum}..{maximum}"
                )
            if not self.context.set_input_shape(self.input_name, target_shape):
                raise RuntimeError(f"TensorRT rejected input shape {target_shape}")
        elif size is not None:
            height, width = (size, size) if isinstance(size, int) else size
            requested = (1, 3, height, width)
            if requested != engine_shape:
                raise ValueError(
                    f"Static engine requires input shape {engine_shape}, got {requested}"
                )
        infer_shapes = getattr(self.context, "infer_shapes", None)
        if infer_shapes is not None:
            unresolved = infer_shapes()
            if unresolved:
                raise RuntimeError(
                    "TensorRT could not resolve tensor shapes; missing inputs: "
                    + ", ".join(str(name) for name in unresolved)
                )
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        self._free_buffers(suppress_errors=False)
        try:
            for name in self.tensor_names:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dimension < 0 for dimension in shape):
                    raise RuntimeError(f"Unresolved TensorRT shape for {name}: {shape}")
                dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
                nbytes = int(np.prod(shape)) * dtype.itemsize
                host_pointer = int(
                    _check_cuda(
                        self.cudart.cudaMallocHost(nbytes),
                        f"cudaMallocHost({name})",
                    )
                )
                device: int | None = None
                try:
                    host = (
                        np.ctypeslib.as_array((ctypes.c_byte * nbytes).from_address(host_pointer))
                        .view(dtype)
                        .reshape(shape)
                    )
                    device = int(_check_cuda(self.cudart.cudaMalloc(nbytes), f"cudaMalloc({name})"))
                    if not self.context.set_tensor_address(name, device):
                        raise RuntimeError(f"TensorRT rejected device address for tensor {name}")
                except Exception:
                    if device is not None:
                        try:
                            _check_cuda(self.cudart.cudaFree(device), f"cudaFree({name})")
                        except Exception:
                            LOGGER.exception("Failed to roll back device allocation for %s", name)
                    try:
                        _check_cuda(self.cudart.cudaFreeHost(host_pointer), f"cudaFreeHost({name})")
                    except Exception:
                        LOGGER.exception("Failed to roll back host allocation for %s", name)
                    raise
                self.buffers[name] = TensorBuffer(name, shape, dtype, host, device)
        except Exception:
            self._free_buffers(suppress_errors=True)
            raise
        input_buffer = self.buffers[self.input_name]
        self.input_shape = input_buffer.shape
        self.input_dtype = input_buffer.dtype

    def prepare(self, source: str | Path | np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        """Prepare a BGR image for the currently selected NCHW engine input."""
        image = load_bgr_image(source)
        if len(self.input_shape) != 4 or self.input_shape[:2] != (1, 3):
            raise ValueError(f"Expected [1,3,H,W] TensorRT input, got {self.input_shape}")
        return preprocess_image(image, self.input_shape[-2:], self.input_dtype)

    def infer_raw(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Copy input, enqueue inference, copy outputs, and synchronize the CUDA stream."""
        self._require_open()
        input_buffer = self.buffers[self.input_name]
        tensor = np.ascontiguousarray(tensor, dtype=input_buffer.dtype)
        if tensor.shape != input_buffer.shape:
            raise ValueError(
                f"Input shape {tensor.shape} does not match engine {input_buffer.shape}"
            )
        np.copyto(input_buffer.host, tensor)
        kind = self.cudart.cudaMemcpyKind
        _check_cuda(
            self.cudart.cudaMemcpyAsync(
                input_buffer.device,
                input_buffer.host.ctypes.data,
                input_buffer.host.nbytes,
                kind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "input cudaMemcpyAsync",
        )
        if not self.context.execute_async_v3(stream_handle=self.stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        for name in self.output_names:
            buffer = self.buffers[name]
            _check_cuda(
                self.cudart.cudaMemcpyAsync(
                    buffer.host.ctypes.data,
                    buffer.device,
                    buffer.host.nbytes,
                    kind.cudaMemcpyDeviceToHost,
                    self.stream,
                ),
                f"output cudaMemcpyAsync({name})",
            )
        self.synchronize()
        return [self.buffers[name].host.copy() for name in self.output_names]

    def synchronize(self) -> None:
        """Synchronize the runner's CUDA stream for correct timing."""
        self._require_open()
        _check_cuda(self.cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")

    def predict(self, source: str | Path | np.ndarray) -> list[Detection]:
        """Run the complete TensorRT YOLO image pipeline."""
        tensor, info = self.prepare(source)
        outputs = self.infer_raw(tensor)
        if not self.names:
            channels = min(np.asarray(outputs[0]).shape[-2:])
            self.names = normalize_names(None, max(0, channels - 4))
        return decode_yolo_output(outputs, info, self.names, self.confidence, self.iou)

    def _free_buffers(self, *, suppress_errors: bool) -> None:
        if self.cudart is None:
            self.buffers.clear()
            return
        errors: list[Exception] = []
        for buffer in list(self.buffers.values()):
            try:
                _check_cuda(self.cudart.cudaFree(buffer.device), f"cudaFree({buffer.name})")
            except Exception as exc:
                errors.append(exc)
            try:
                _check_cuda(
                    self.cudart.cudaFreeHost(buffer.host.ctypes.data),
                    f"cudaFreeHost({buffer.name})",
                )
            except Exception as exc:
                errors.append(exc)
        self.buffers.clear()
        if errors and not suppress_errors:
            raise RuntimeError(f"Failed to release {len(errors)} TensorRT buffer allocation(s)")

    def close(self, *, suppress_errors: bool = False) -> None:
        """Release CUDA allocations before context, engine, and runtime references."""
        if self._closed:
            return
        errors: list[Exception] = []
        try:
            self._free_buffers(suppress_errors=False)
        except Exception as exc:
            errors.append(exc)
        if self.cudart is not None and self.stream is not None:
            try:
                _check_cuda(self.cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
            except Exception as exc:
                errors.append(exc)
        self.stream = None
        self.context = None
        self.engine = None
        self.runtime = None
        self._closed = True
        if errors and not suppress_errors:
            raise RuntimeError(f"Failed to release {len(errors)} TensorRT resource(s)")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("TensorRT detector is closed")

    def __enter__(self) -> TensorRTDetector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed") and not self._closed:
            self.close(suppress_errors=True)
