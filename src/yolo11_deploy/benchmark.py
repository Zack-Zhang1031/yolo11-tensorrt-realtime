"""Synchronized latency measurement shared by every inference backend."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Standard benchmark parameters."""

    backend: str
    precision: str
    input_size: int | tuple[int, int] = 640
    batch: int = 1
    warmup: int = 50
    runs: int = 200

    def __post_init__(self) -> None:
        dimensions = (
            (self.input_size, self.input_size)
            if isinstance(self.input_size, int)
            else self.input_size
        )
        if any(dimension <= 0 for dimension in dimensions) or self.batch <= 0:
            raise ValueError("input_size and batch must be positive")
        if self.warmup < 0 or self.runs <= 0:
            raise ValueError("warmup must be non-negative and runs must be positive")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Latency summary in milliseconds and throughput in frames per second."""

    backend: str
    precision: str
    input_size: int | tuple[int, int]
    batch: int
    warmup: int
    runs: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    fps: float

    def format_row(self) -> str:
        """Format one human-readable benchmark table row."""
        return (
            f"{self.backend + ' ' + self.precision:<25}"
            f"{self.mean_latency_ms:>12.3f}"
            f"{self.median_latency_ms:>10.3f}"
            f"{self.p95_latency_ms:>10.3f}"
            f"{self.fps:>10.2f}"
        )


def run_benchmark(
    inference: Callable[[], object],
    config: BenchmarkConfig,
    synchronize: Callable[[], None] | None = None,
) -> BenchmarkResult:
    """Benchmark a callable, synchronizing asynchronous devices before timestamps."""
    sync = synchronize or (lambda: None)
    for _ in range(config.warmup):
        inference()
    sync()

    latencies_ms: list[float] = []
    for _ in range(config.runs):
        sync()
        start_ns = time.perf_counter_ns()
        inference()
        sync()
        latencies_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000)

    mean_latency = statistics.fmean(latencies_ms)
    return BenchmarkResult(
        backend=config.backend,
        precision=config.precision,
        input_size=config.input_size,
        batch=config.batch,
        warmup=config.warmup,
        runs=config.runs,
        mean_latency_ms=mean_latency,
        median_latency_ms=statistics.median(latencies_ms),
        p95_latency_ms=float(np.percentile(latencies_ms, 95)),
        fps=config.batch * 1000.0 / mean_latency,
    )


def print_benchmark(result: BenchmarkResult) -> None:
    """Print a consistent benchmark table."""
    print(f"{'Backend':<25}{'Latency(ms)':>12}{'P50':>10}{'P95':>10}{'FPS':>10}")
    print(result.format_row())
