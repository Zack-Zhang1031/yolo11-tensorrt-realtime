from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import yolo11_deploy.video as video


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self) -> None:
        self.released = True


class FakeWriter:
    def __init__(self) -> None:
        self.frames = 0
        self.released = False

    def write(self, _: np.ndarray) -> None:
        self.frames += 1

    def release(self) -> None:
        self.released = True

    def isOpened(self) -> bool:
        return True


def test_process_capture_releases_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = FakeCapture([np.zeros((8, 8, 3), dtype=np.uint8)])
    writer = FakeWriter()
    detector = SimpleNamespace(predict=lambda _: [])
    destroyed: list[bool] = []
    monkeypatch.setattr(video, "draw_detections", lambda frame, *_: frame)
    monkeypatch.setattr(video.cv2, "destroyAllWindows", lambda: destroyed.append(True))
    count = video.process_capture(
        capture,
        detector,
        window_name="test",
        writer=writer,
        display=False,
    )
    assert count == 1
    assert capture.released and writer.released and destroyed


def test_process_capture_releases_after_inference_error(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = FakeCapture([np.zeros((8, 8, 3), dtype=np.uint8)])
    detector = SimpleNamespace(predict=lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(video.cv2, "destroyAllWindows", lambda: None)
    with pytest.raises(RuntimeError, match="boom"):
        video.process_capture(capture, detector, window_name="test", display=False)
    assert capture.released


def test_empty_capture_returns_zero_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = FakeCapture([])
    detector = SimpleNamespace(predict=lambda _: [])
    monkeypatch.setattr(video.cv2, "destroyAllWindows", lambda: None)
    count = video.process_capture(capture, detector, window_name="test", display=False)
    assert count == 0
    assert capture.released


@pytest.mark.parametrize(("suffix", "codec"), [(".mp4", "mp4v"), (".avi", "MJPG")])
def test_writer_codec_matches_extension(
    suffix: str,
    codec: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    writer = FakeWriter()
    seen: list[str] = []
    monkeypatch.setattr(
        video.cv2,
        "VideoWriter_fourcc",
        lambda *chars: seen.append("".join(chars)) or 1,
    )
    monkeypatch.setattr(video.cv2, "VideoWriter", lambda *_: writer)
    result, output = video.create_video_writer(tmp_path / f"output{suffix}", (8, 8), 30.0)
    assert result is writer
    assert output.suffix == suffix
    assert seen == [codec]
