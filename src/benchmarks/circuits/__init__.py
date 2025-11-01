"""Logical benchmark circuit subclasses."""

from .bell import BellStateBenchmark
from .ghz import GHZ3Benchmark
from .parity_check import ParityCheckBenchmark
from .simple import Simple1QXZHBenchmark

__all__ = [
    "BellStateBenchmark",
    "GHZ3Benchmark",
    "ParityCheckBenchmark",
    "Simple1QXZHBenchmark",
]
