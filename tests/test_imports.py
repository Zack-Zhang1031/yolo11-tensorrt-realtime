import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "yolo11_deploy",
        "yolo11_deploy.detector",
        "yolo11_deploy.preprocessing",
        "yolo11_deploy.postprocessing",
        "yolo11_deploy.visualization",
        "yolo11_deploy.benchmark",
        "yolo11_deploy.onnx_runtime",
        "yolo11_deploy.tensorrt_runtime",
    ],
)
def test_package_modules_import_without_optional_tensorrt(module: str) -> None:
    assert importlib.import_module(module) is not None

