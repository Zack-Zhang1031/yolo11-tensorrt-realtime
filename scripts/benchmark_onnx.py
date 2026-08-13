"""Benchmark raw YOLO11 ONNX Runtime inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from yolo11_deploy.benchmark import BenchmarkConfig, print_benchmark, run_benchmark
from yolo11_deploy.onnx_runtime import ONNXDetector
from yolo11_deploy.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--imgsz", type=int, default=640, help="Fallback size for dynamic ONNX")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    detector = ONNXDetector(args.model, device=args.device, image_size=args.imgsz)
    height, width = detector.input_size
    dtype = np.float16 if detector.input.type == "tensor(float16)" else np.float32
    tensor = np.zeros((1, 3, height, width), dtype=dtype)
    result = run_benchmark(
        lambda: detector.infer_raw(tensor),
        BenchmarkConfig(
            "ONNX Runtime",
            "FP16" if dtype == np.float16 else "FP32",
            height,
            1,
            args.warmup,
            args.runs,
        ),
    )
    print_benchmark(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
