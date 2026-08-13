"""TensorRT 10.x engine construction from static or dynamic ONNX models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tensorrt_runtime import require_cuda_device
from .utils import ensure_parent, optional_dependency_available


@dataclass(frozen=True, slots=True)
class OptimizationProfile:
    """Square image-size range for a dynamic TensorRT optimization profile."""

    minimum: int = 320
    optimum: int = 640
    maximum: int = 1280

    def __post_init__(self) -> None:
        if not 0 < self.minimum <= self.optimum <= self.maximum:
            raise ValueError("Profile sizes must satisfy 0 < minimum <= optimum <= maximum")


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    fp16: bool = False,
    workspace_gib: float = 4.0,
    profile: OptimizationProfile | None = None,
) -> Path:
    """Parse ONNX, build a TensorRT 10.x engine, and preserve class metadata."""
    if workspace_gib <= 0:
        raise ValueError("workspace_gib must be positive")
    if not optional_dependency_available("tensorrt"):
        raise RuntimeError("TensorRT 10.x and an NVIDIA CUDA environment are required")

    import tensorrt as trt  # pyright: ignore[reportMissingImports]

    if int(trt.__version__.split(".", maxsplit=1)[0]) != 10:
        raise RuntimeError(f"This builder supports TensorRT 10.x; found {trt.__version__}")
    _, device_count = require_cuda_device()
    print(f"TensorRT version: {trt.__version__}")
    print(f"CUDA devices: {device_count}")

    source = Path(onnx_path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Non-empty ONNX file not found: {source}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(source.read_bytes()):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parsing failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 1024**3))
    _configure_dynamic_inputs(builder, network, config, profile or OptimizationProfile())
    if fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("FP16 requested, but the TensorRT platform lacks fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build a serialized engine")
    target = ensure_parent(engine_path)
    target.write_bytes(bytes(serialized))
    if target.stat().st_size == 0:
        raise RuntimeError(f"TensorRT wrote an empty engine: {target}")

    metadata = read_onnx_metadata(source)
    if metadata:
        metadata_path = target.with_suffix(f"{target.suffix}.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Metadata: {metadata_path}")
    print(f"Engine: {target}")
    print(f"Size: {target.stat().st_size / (1024 * 1024):.2f} MiB")
    return target


def _configure_dynamic_inputs(
    builder: Any,
    network: Any,
    config: Any,
    sizes: OptimizationProfile,
) -> None:
    dynamic_inputs = [
        network.get_input(index)
        for index in range(network.num_inputs)
        if any(dimension < 0 for dimension in network.get_input(index).shape)
    ]
    if not dynamic_inputs:
        return
    profile = builder.create_optimization_profile()
    for tensor in dynamic_inputs:
        shape = tuple(tensor.shape)
        if len(shape) != 4 or shape[1] not in {-1, 3}:
            raise ValueError(f"Unsupported dynamic input shape for {tensor.name}: {shape}")
        accepted = profile.set_shape(
            tensor.name,
            (1, 3, sizes.minimum, sizes.minimum),
            (1, 3, sizes.optimum, sizes.optimum),
            (1, 3, sizes.maximum, sizes.maximum),
        )
        if accepted is False:
            raise RuntimeError(f"TensorRT rejected optimization profile for {tensor.name}")
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("TensorRT rejected the optimization profile")


def read_onnx_metadata(onnx_path: Path) -> dict[str, str]:
    """Read lightweight string metadata needed by the inference adapter."""
    try:
        import onnx
    except ImportError:
        return {}
    model = onnx.load(str(onnx_path), load_external_data=False)
    return {entry.key: entry.value for entry in model.metadata_props if entry.key == "names"}
