"""Simulation entry points for logical-error benchmarking."""

from .runner import (
    MonteCarloConfig,
    SimulationResult,
    run_circuit_logical_error_rate,
    run_logical_error_rate,
)

__all__ = [
    "MonteCarloConfig",
    "SimulationResult",
    "run_circuit_logical_error_rate",
    "run_logical_error_rate",
]
