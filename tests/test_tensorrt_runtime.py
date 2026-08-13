import pytest

import yolo11_deploy.tensorrt_runtime as runtime


class FakeCudaRuntime:
    def __init__(self, device_count: int) -> None:
        self.device_count = device_count

    def cudaGetDeviceCount(self) -> tuple[int, int]:
        return 0, self.device_count


def test_require_cuda_device_returns_runtime_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeCudaRuntime(2)
    monkeypatch.setattr(runtime, "_load_cuda_runtime", lambda: fake)
    assert runtime.require_cuda_device() == (fake, 2)


def test_require_cuda_device_rejects_no_visible_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_load_cuda_runtime", lambda: FakeCudaRuntime(0))
    with pytest.raises(RuntimeError, match="visible NVIDIA CUDA device"):
        runtime.require_cuda_device()


def test_check_cuda_reports_operation() -> None:
    with pytest.raises(RuntimeError, match="cudaMalloc"):
        runtime._check_cuda((1,), "cudaMalloc")
