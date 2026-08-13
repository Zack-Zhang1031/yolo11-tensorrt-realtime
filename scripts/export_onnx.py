"""Export an Ultralytics YOLO11 model to ONNX and validate the artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from yolo11_deploy.onnx_validation import validate_onnx_artifact
from yolo11_deploy.utils import configure_logging, ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int)
    parser.add_argument("--dynamic", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    if args.half and args.device == "cpu":
        raise ValueError("--half requires a CUDA device for Ultralytics ONNX export")
    try:
        from ultralytics import YOLO  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError("Install Ultralytics from requirements.txt before ONNX export") from exc

    model = YOLO(args.model)
    export_options: dict[str, object] = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "half": args.half,
        "device": args.device,
        "dynamic": args.dynamic,
        "simplify": False,
    }
    if args.opset is not None:
        export_options["opset"] = args.opset
    exported = Path(model.export(**export_options)).resolve()
    if not exported.is_file() or exported.stat().st_size == 0:
        raise RuntimeError(f"Ultralytics did not produce a non-empty ONNX file: {exported}")
    final_path = exported
    if args.output is not None:
        final_path = ensure_parent(args.output)
        if final_path != exported:
            exported.replace(final_path)
    info = validate_onnx_artifact(final_path)
    print(f"ONNX export: {info.path}")
    print(f"Size: {info.size_bytes / (1024 * 1024):.2f} MiB")
    print(f"Opset: {info.opset}")
    print(f"Inputs: {info.inputs}")
    print(f"Outputs: {info.outputs}")
    class_count = info.class_count if info.class_count is not None else "metadata unavailable"
    print(f"Classes: {class_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
