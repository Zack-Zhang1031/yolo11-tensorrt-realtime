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
