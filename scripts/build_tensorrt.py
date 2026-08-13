"""Build a TensorRT 10.x engine from ONNX with optional FP16 precision."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from yolo11_deploy.utils import configure_logging, ensure_parent, optional_dependency_available


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace", type=float, default=4.0, help="Workspace in GiB")
    return parser.parse_args()


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool, workspace_gib: float) -> Path:
    """Parse ONNX, configure TensorRT, build and persist a serialized engine."""
    if workspace_gib <= 0:
        raise ValueError("workspace must be positive")
    if not optional_dependency_available("tensorrt"):
        raise RuntimeError("TensorRT is optional and requires NVIDIA GPU/CUDA/TensorRT.")
    import tensorrt as trt

    print(f"TensorRT version: {trt.__version__}")
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
    if fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("--fp16 requested, but the current TensorRT platform lacks fast FP16")
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
    return target


def main() -> int:
    args = parse_args()
    configure_logging()
    build_engine(args.onnx, args.engine, args.fp16, args.workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

