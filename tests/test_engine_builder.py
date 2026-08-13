from types import SimpleNamespace

import pytest

from yolo11_deploy.engine_builder import OptimizationProfile, _configure_dynamic_inputs


def test_profile_validates_order() -> None:
    with pytest.raises(ValueError, match="minimum"):
        OptimizationProfile(640, 320, 1280)


def test_dynamic_profile_is_added() -> None:
    calls: list[tuple[object, ...]] = []
    profile = SimpleNamespace(set_shape=lambda *args: calls.append(args) or True)
    builder = SimpleNamespace(create_optimization_profile=lambda: profile)
    dynamic_input = SimpleNamespace(name="images", shape=(-1, 3, -1, -1))
    network = SimpleNamespace(num_inputs=1, get_input=lambda _: dynamic_input)
    config = SimpleNamespace(add_optimization_profile=lambda value: 0 if value is profile else -1)
    _configure_dynamic_inputs(builder, network, config, OptimizationProfile(320, 640, 1280))
    assert calls == [
        (
            "images",
            (1, 3, 320, 320),
            (1, 3, 640, 640),
            (1, 3, 1280, 1280),
        )
    ]
