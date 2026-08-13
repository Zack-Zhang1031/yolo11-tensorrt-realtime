import sys
from types import SimpleNamespace

import pytest

from yolo11_deploy.onnx_runtime import ONNXDetector


class FakeInput:
    def __init__(self, shape: list[object]) -> None:
        self.shape = shape


@pytest.mark.parametrize(
    ("shape", "fallback", "expected"),
    [
        ([1, 3, 640, 640], 320, (640, 640)),
        (["batch", 3, "height", "width"], 640, (640, 640)),
        ([1, 3, "height", "width"], (384, 672), (384, 672)),
    ],
)
def test_input_size_resolves_static_and_dynamic_shapes(
    shape: list[object],
    fallback: int | tuple[int, int],
    expected: tuple[int, int],
) -> None:
    detector = object.__new__(ONNXDetector)
    detector.input = FakeInput(shape)
    assert detector._input_size(fallback) == expected


def test_input_size_rejects_invalid_rank() -> None:
    detector = object.__new__(ONNXDetector)
    detector.input = FakeInput([1, 640, 640])
    with pytest.raises(ValueError, match="NCHW"):
        detector._input_size(640)


class FakeSession:
    def __init__(self, providers: list[object], active: list[str]) -> None:
        self.requested_providers = providers
        self.active = active
        self.fallback_disabled = False

    def get_inputs(self) -> list[object]:
        return [SimpleNamespace(name="images", shape=[1, 3, 640, 640], type="tensor(float)")]

    def get_outputs(self) -> list[object]:
        return [SimpleNamespace(name="output0")]

    def get_providers(self) -> list[str]:
        return self.active

    def get_modelmeta(self) -> object:
        return SimpleNamespace(custom_metadata_map={"names": "{0: 'object'}"})

    def disable_fallback(self) -> None:
        self.fallback_disabled = True


def _fake_ort(active: list[str]) -> object:
    module = SimpleNamespace()
    module.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    module.preload_dlls = lambda: None
    module.session = None

    def create_session(_: str, providers: list[object]) -> FakeSession:
        module.session = FakeSession(providers, active)
        return module.session

    module.InferenceSession = create_session
    return module


def test_explicit_cuda_disables_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    fake_ort = _fake_ort(["CUDAExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    detector = ONNXDetector(model_path, device="cuda", device_id=2)
    assert detector.providers == ["CUDAExecutionProvider"]
    assert detector.device_id == 2
    assert fake_ort.session.fallback_disabled
    assert fake_ort.session.requested_providers == [("CUDAExecutionProvider", {"device_id": 2})]


def test_explicit_cuda_rejects_provider_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort(["CPUExecutionProvider"]))
    with pytest.raises(RuntimeError, match="not activated"):
        ONNXDetector(model_path, device="cuda")
