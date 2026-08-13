"""Benchmark raw TensorRT YOLO11 inference with CUDA stream synchronization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from yolo11_deploy.benchmark import BenchmarkConfig, print_benchmark, run_benchmark
from yolo11_deploy.tensorrt_runtime import TensorRTDetector
from yolo11_deploy.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    with TensorRTDetector(args.engine) as detector:
        tensor = np.zeros(detector.input_shape, dtype=detector.input_dtype)
        result = run_benchmark(
            lambda: detector.infer_raw(tensor),
            BenchmarkConfig(
                "TensorRT",
                "FP16" if detector.input_dtype == np.float16 else str(detector.input_dtype),
                detector.input_shape[-1],
                1,
                args.warmup,
                args.runs,
            ),
            detector.synchronize,
        )
    print_benchmark(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
