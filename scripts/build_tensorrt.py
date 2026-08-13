"""Build a TensorRT 10.x engine from ONNX with optional FP16 precision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo11_deploy.tensorrt_runtime import require_cuda_device
from yolo11_deploy.utils import configure_logging, ensure_parent, optional_dependency_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace", type=float, default=4.0, help="Workspace in GiB")
    parser.add_argument("--min-imgsz", type=int, default=320)
    parser.add_argument("--opt-imgsz", type=int, default=640)
    parser.add_argument("--max-imgsz", type=int, default=1280)
    return parser.parse_args()


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool,
    workspace_gib: float,
    min_imgsz: int = 320,
    opt_imgsz: int = 640,
    max_imgsz: int = 1280,
) -> Path:
    """Parse ONNX, configure TensorRT, build and persist a serialized engine."""
    if workspace_gib <= 0:
        raise ValueError("workspace must be positive")
    if not optional_dependency_available("tensorrt"):
        raise RuntimeError("TensorRT is optional and requires NVIDIA GPU/CUDA/TensorRT.")
    import tensorrt as trt

    _, device_count = require_cuda_device()
    print(f"TensorRT version: {trt.__version__}")
    print(f"CUDA devices: {device_count}")
    onnx_path = onnx_path.expanduser().resolve()
    if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
        raise FileNotFoundError(f"Non-empty ONNX file not found: {onnx_path}")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parsing failed:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 1024**3))
    dynamic_inputs = [
        network.get_input(index)
        for index in range(network.num_inputs)
        if any(dimension < 0 for dimension in network.get_input(index).shape)
    ]
    if dynamic_inputs:
        if not 0 < min_imgsz <= opt_imgsz <= max_imgsz:
            raise ValueError("Dynamic profile sizes must satisfy 0 < min <= opt <= max")
        profile = builder.create_optimization_profile()
        for tensor in dynamic_inputs:
            shape = tuple(tensor.shape)
            if len(shape) != 4 or shape[1] not in {-1, 3}:
                raise ValueError(f"Unsupported dynamic input shape for {tensor.name}: {shape}")
            profile.set_shape(
                tensor.name,
                (1, 3, min_imgsz, min_imgsz),
                (1, 3, opt_imgsz, opt_imgsz),
                (1, 3, max_imgsz, max_imgsz),
            )
        config.add_optimization_profile(profile)
    if fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError(
                "--fp16 requested, but the current TensorRT platform lacks fast FP16"
            )
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build a serialized engine")
    target = ensure_parent(engine_path)
    target.write_bytes(bytes(serialized))
    if target.stat().st_size == 0:
        raise RuntimeError(f"TensorRT wrote an empty engine: {target}")
    print(f"Engine: {target}")
    print(f"Size: {target.stat().st_size / (1024 * 1024):.2f} MiB")
    metadata = _read_onnx_metadata(onnx_path)
    if metadata:
        metadata_path = target.with_suffix(f"{target.suffix}.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Metadata: {metadata_path}")
    return target


def _read_onnx_metadata(onnx_path: Path) -> dict[str, str]:
    """Read lightweight string metadata required by the TensorRT result adapter."""
    try:
        import onnx
    except ImportError:
        return {}
    model = onnx.load(str(onnx_path), load_external_data=False)
    return {entry.key: entry.value for entry in model.metadata_props if entry.key == "names"}


def main() -> int:
    args = parse_args()
    configure_logging()
    build_engine(
        args.onnx,
        args.engine,
        args.fp16,
        args.workspace,
        args.min_imgsz,
        args.opt_imgsz,
        args.max_imgsz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
