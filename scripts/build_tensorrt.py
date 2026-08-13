"""Build a TensorRT 10.x engine from ONNX with optional FP16 precision."""

from __future__ import annotations

import argparse
from pathlib import Path

from yolo11_deploy.engine_builder import OptimizationProfile, build_engine
from yolo11_deploy.utils import configure_logging


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


def main() -> int:
    args = parse_args()
    configure_logging()
    profile = OptimizationProfile(args.min_imgsz, args.opt_imgsz, args.max_imgsz)
    build_engine(
        args.onnx,
        args.engine,
        fp16=args.fp16,
        workspace_gib=args.workspace,
        profile=profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
