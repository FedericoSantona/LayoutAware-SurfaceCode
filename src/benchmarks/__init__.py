"""Benchmark circuit package exposing logical templates."""

from .BenchmarkCircuit import BenchmarkCircuit
from .circuits import (
    BellStateBenchmark,
    GHZ3Benchmark,
    ParityCheckBenchmark,
)

__all__ = [
    "BenchmarkCircuit",
    "BellStateBenchmark",
    "GHZ3Benchmark",
    "ParityCheckBenchmark",
]
