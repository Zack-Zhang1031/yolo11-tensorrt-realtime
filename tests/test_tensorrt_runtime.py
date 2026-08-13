import ctypes
from types import SimpleNamespace

import numpy as np
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


class TrackingCudaRuntime:
    def __init__(self, fail_device: int | None = None) -> None:
        self.fail_device = fail_device
        self.freed_devices: list[int] = []
        self.freed_hosts: list[int] = []
        self.destroyed_streams: list[int] = []

    def cudaFree(self, pointer: int) -> tuple[int]:
        self.freed_devices.append(pointer)
        return (1 if pointer == self.fail_device else 0,)

    def cudaFreeHost(self, pointer: int) -> tuple[int]:
        self.freed_hosts.append(pointer)
        return (0,)

    def cudaStreamDestroy(self, stream: int) -> tuple[int]:
        self.destroyed_streams.append(stream)
        return (0,)


def test_initialization_failure_releases_created_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = TrackingCudaRuntime()

    def fail_after_stream(self: runtime.TensorRTDetector, *_: object) -> None:
        self.cudart = cuda
        self.stream = 73
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(runtime.TensorRTDetector, "_initialize", fail_after_stream)
    with pytest.raises(RuntimeError, match="initialization failed"):
        runtime.TensorRTDetector("missing.engine")
    assert cuda.destroyed_streams == [73]


def test_close_continues_after_one_buffer_release_failure() -> None:
    detector = object.__new__(runtime.TensorRTDetector)
    cuda = TrackingCudaRuntime(fail_device=11)
    first_host = np.zeros(1, dtype=np.uint8)
    second_host = np.zeros(1, dtype=np.uint8)
    detector._closed = False
    detector.cudart = cuda
    detector.stream = 99
    detector.context = SimpleNamespace()
    detector.engine = SimpleNamespace()
    detector.runtime = SimpleNamespace()
    detector.buffers = {
        "first": runtime.TensorBuffer("first", (1,), first_host.dtype, first_host, 11),
        "second": runtime.TensorBuffer("second", (1,), second_host.dtype, second_host, 22),
    }

    detector.close(suppress_errors=True)

    assert cuda.freed_devices == [11, 22]
    assert cuda.freed_hosts == [first_host.ctypes.data, second_host.ctypes.data]
    assert cuda.destroyed_streams == [99]
    assert detector.buffers == {}
    assert detector._closed


def test_address_rejection_rolls_back_host_and_device_allocations() -> None:
    detector = object.__new__(runtime.TensorRTDetector)
    cuda = TrackingCudaRuntime()
    pinned = ctypes.create_string_buffer(16)
    cuda.cudaMallocHost = lambda _: (0, ctypes.addressof(pinned))  # type: ignore[attr-defined]
    cuda.cudaMalloc = lambda _: (0, 321)  # type: ignore[attr-defined]
    detector._closed = False
    detector.cudart = cuda
    detector.stream = None
    detector.runtime = None
    detector.trt = SimpleNamespace(nptype=lambda _: np.float32)
    detector.engine = SimpleNamespace(get_tensor_dtype=lambda _: object())
    detector.context = SimpleNamespace(
        get_tensor_shape=lambda _: (1, 4),
        set_tensor_address=lambda *_: False,
    )
    detector.tensor_names = ["output"]
    detector.input_name = "output"
    detector.buffers = {}

    with pytest.raises(RuntimeError, match="rejected device address"):
        detector._allocate_buffers()

    assert cuda.freed_devices == [321]
    assert cuda.freed_hosts == [ctypes.addressof(pinned)]
