"""Run fast environment and end-to-end YOLO11 deployment smoke checks."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable

import numpy as np


def report(status: str, label: str, detail: str = "") -> None:
    suffix = f": {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def check(label: str, operation: Callable[[], object]) -> bool:
    try:
        operation()
    except Exception as exc:
        report("FAIL", label, f"{type(exc).__name__}: {exc}")
        return False
    report("PASS", label)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Skip weight download and inference")
    args = parser.parse_args()
    failures = 0
    try:
        import yolo11_deploy
        from yolo11_deploy.benchmark import BenchmarkConfig, run_benchmark
        from yolo11_deploy.detector import YOLODetector
        from yolo11_deploy.postprocessing import decode_yolo_output
        from yolo11_deploy.preprocessing import preprocess_image

        assert yolo11_deploy.__name__ == "yolo11_deploy"
    except Exception as exc:
        report("FAIL", "package import", f"{type(exc).__name__}: {exc}")
        return 1
    report("PASS", "package import")

    try:
        import torch

        report("PASS", "torch available", torch.__version__)
    except ImportError as exc:
        report("FAIL", "torch available", str(exc))
        return 1

    try:
        import ultralytics  # pyright: ignore[reportMissingImports]

        report("PASS", "ultralytics import", ultralytics.__version__)
    except ImportError as exc:
        report("FAIL", "ultralytics import", str(exc))
        return 1

    synthetic = np.zeros((480, 640, 3), dtype=np.uint8)
    preprocessed: tuple[np.ndarray, object] | None = None

    def run_preprocessing() -> None:
        nonlocal preprocessed
        preprocessed = preprocess_image(synthetic, 640)
        assert preprocessed[0].shape == (1, 3, 640, 640)

    failures += not check("synthetic image preprocessing", run_preprocessing)

    detector: YOLODetector | None = None

    def load_model() -> None:
        nonlocal detector
        detector = YOLODetector("yolo11s.pt", device="cpu", image_size=640)

    loaded = False
    if args.offline:
        report("SKIP", "YOLO11s model build/load", "offline mode")
        report("SKIP", "PyTorch inference pipeline", "offline mode")
    else:
        loaded = check("YOLO11s model build/load", load_model)
        failures += not loaded
        if loaded and detector is not None:
            active_detector = detector
            failures += not check(
                "PyTorch inference pipeline", lambda: active_detector.predict(synthetic)
            )
        else:
            report("SKIP", "PyTorch inference pipeline", "model load failed")

    def run_postprocessing() -> None:
        from yolo11_deploy.preprocessing import LetterboxInfo

        output = np.zeros((1, 5, 1), dtype=np.float32)
        output[0, :, 0] = [320, 320, 100, 100, 0.9]
        detections = decode_yolo_output(
            output,
            LetterboxInfo((640, 640), (640, 640), 1.0, 0.0, 0.0),
            {0: "object"},
        )
        assert len(detections) == 1

    failures += not check("postprocessing", run_postprocessing)

    def run_benchmark_utility() -> None:
        result = run_benchmark(lambda: 1 + 1, BenchmarkConfig("Smoke", "N/A", 32, 1, 1, 3))
        assert result.runs == 3 and result.mean_latency_ms >= 0

    failures += not check("benchmark utilities", run_benchmark_utility)

    if importlib.util.find_spec("onnxruntime"):
        import onnxruntime as ort

        report("PASS", "ONNX Runtime availability", ", ".join(ort.get_available_providers()))
    else:
        report("SKIP", "ONNX Runtime availability", "optional dependency not installed")

    if torch.cuda.is_available():
        report("PASS", "CUDA", torch.cuda.get_device_name(0))
    else:
        report("SKIP", "CUDA", "no CUDA device visible to PyTorch")

    if importlib.util.find_spec("tensorrt"):
        import tensorrt as trt  # pyright: ignore[reportMissingImports]

        report("PASS", "TensorRT", trt.__version__)
    else:
        report("SKIP", "TensorRT optional dependency", "not installed")

    if failures:
        report("FAIL", "smoke test summary", f"{failures} required check(s) failed")
        return 1
    report("PASS", "smoke test summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
