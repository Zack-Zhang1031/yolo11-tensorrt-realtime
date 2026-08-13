import pytest

from yolo11_deploy.benchmark import BenchmarkConfig, run_benchmark


def test_benchmark_runs_warmups_measurements_and_syncs() -> None:
    calls = {"infer": 0, "sync": 0}

    def inference() -> None:
        calls["infer"] += 1

    def synchronize() -> None:
        calls["sync"] += 1

    result = run_benchmark(
        inference,
        BenchmarkConfig("Test", "FP32", warmup=2, runs=3),
        synchronize,
    )
    assert calls == {"infer": 5, "sync": 7}
    assert result.runs == 3
    assert result.mean_latency_ms >= 0
    assert result.fps > 0


def test_benchmark_config_rejects_no_measurements() -> None:
    with pytest.raises(ValueError, match="runs"):
        BenchmarkConfig("Test", "FP32", runs=0)

