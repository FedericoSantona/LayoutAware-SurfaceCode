"""Simulation entry points for logical-error benchmarking."""

from .ler_runner import (
    SimulationResult,
    run_circuit_logical_error_rate,
    run_cnot_logical_error_rate,
    run_logical_error_rate,
)
from .physics_runner import (
    PhysicsResult,
    run_circuit_physics,
)
from .montecarlo import MonteCarloConfig
__all__ = [
    "SimulationResult",
    "run_circuit_logical_error_rate",
    "run_cnot_logical_error_rate",
    "run_logical_error_rate",
    "PhysicsResult",
    "run_circuit_physics",
    "MonteCarloConfig",
]
