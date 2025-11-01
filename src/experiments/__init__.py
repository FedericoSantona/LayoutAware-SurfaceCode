"""Experiments entry points for logical-error benchmarking."""

from .code_threshold.threshold import (
    MonteCarloConfig,
    SimulationResult,
    run_logical_error_rate,
)

__all__ = [
    "MonteCarloConfig",
    "SimulationResult", 
    "run_logical_error_rate",
]
