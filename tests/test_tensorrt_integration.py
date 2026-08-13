import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.tensorrt
def test_build_and_run_tensorrt_engine(tmp_path: Path) -> None:
    if os.environ.get("YOLO_TRT_INTEGRATION") != "1":
        pytest.skip("set YOLO_TRT_INTEGRATION=1 to enable the TensorRT integration test")
    if importlib.util.find_spec("tensorrt") is None:
        pytest.skip("TensorRT is not installed")
    onnx_path = Path(os.environ.get("YOLO_TRT_ONNX", "weights/yolo11s.onnx"))
    if not onnx_path.is_file():
        pytest.skip(f"ONNX model is unavailable: {onnx_path}")

    from yolo11_deploy.engine_builder import build_engine
    from yolo11_deploy.tensorrt_runtime import TensorRTDetector

    engine_path = build_engine(onnx_path, tmp_path / "yolo11s.engine", fp16=True)
    with TensorRTDetector(engine_path, image_size=640) as detector:
        output = detector.infer_raw(np.zeros(detector.input_shape, dtype=detector.input_dtype))
    assert output
    assert output[0].size > 0
