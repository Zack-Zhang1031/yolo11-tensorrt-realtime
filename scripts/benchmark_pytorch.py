"""Benchmark raw YOLO11 PyTorch forward inference with correct CUDA synchronization."""

from __future__ import annotations

import argparse

import torch

from yolo11_deploy.benchmark import BenchmarkConfig, print_benchmark, run_benchmark
from yolo11_deploy.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    if args.half and not args.device.startswith("cuda"):
        raise ValueError("PyTorch FP16 benchmark requires --device cuda:0 (or another CUDA device)")
    from ultralytics import YOLO

    yolo = YOLO(args.model)
    module = yolo.model.to(args.device).eval()
    dtype = torch.float16 if args.half else torch.float32
    if args.half:
        module.half()
    tensor = torch.zeros((1, 3, args.imgsz, args.imgsz), dtype=dtype, device=args.device)
    synchronize = torch.cuda.synchronize if args.device.startswith("cuda") else None
    config = BenchmarkConfig(
        backend="PyTorch",
        precision="FP16" if args.half else "FP32",
        input_size=args.imgsz,
        warmup=args.warmup,
        runs=args.runs,
    )
    with torch.inference_mode():
        result = run_benchmark(lambda: module(tensor), config, synchronize)
    print_benchmark(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

