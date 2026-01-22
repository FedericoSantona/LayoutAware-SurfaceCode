"""Threshold sweep utilities for surface-code logical error studies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from surface_code import (
    PhenomenologicalStimBuilder,
    PhenomenologicalStimConfig,
    build_surface_code_model,
)

if TYPE_CHECKING:
    from surface_code.noise_model import NoiseModel

from ..ler_runner import SimulationResult, run_logical_error_rate, run_cnot_logical_error_rate
from ..montecarlo import MonteCarloConfig

# Supported experiment types
EXPERIMENT_TYPE_MEMORY = "memory"
EXPERIMENT_TYPE_CNOT = "cnot"


@dataclass
class ThresholdPoint:
    p_x: float
    p_z: float
    logical_error_rate: float
    avg_syndrome_weight: float
    click_rate: float


@dataclass
class DistanceSweepResult:
    distance: int
    points: List[ThresholdPoint]


@dataclass
class ThresholdScenarioResult:
    name: str
    init_label: str
    track: str
    distances: List[int]
    sweeps: List[DistanceSweepResult]

    def as_numpy(self) -> Dict[int, np.ndarray]:
        return {res.distance: np.array([(pt.p_x, pt.p_z, pt.logical_error_rate) for pt in res.points]) for res in self.sweeps}


@dataclass
class ThresholdScenario:
    name: str
    init_label: str
    track: str  # 'Z' or 'X'
    distances: Sequence[int]
    physical_error_grid: Sequence[float]
    rounds_scale: float = 1.0

    def rounds_for_distance(self, distance: int) -> int:
        return max(1, int(round(self.rounds_scale * distance)))

    def px_pz_pairs(self) -> Iterable[tuple[float, float]]:
        raise NotImplementedError


class XOnlyScenario(ThresholdScenario):
    def px_pz_pairs(self) -> Iterable[tuple[float, float]]:
        for p in self.physical_error_grid:
            yield p, 0.0


class ZOnlyScenario(ThresholdScenario):
    def px_pz_pairs(self) -> Iterable[tuple[float, float]]:
        for p in self.physical_error_grid:
            yield 0.0, p


class SymmetricScenario(ThresholdScenario):
    def px_pz_pairs(self) -> Iterable[tuple[float, float]]:
        for p in self.physical_error_grid:
            yield p, p


@dataclass
class ThresholdStudyConfig:
    shots: int = 5000
    seed: int | None = None


def run_scenario(
    scenario: ThresholdScenario,
    study_cfg: ThresholdStudyConfig,
    progress: Callable[[ThresholdScenario, int, float, float], None] | None = None,
    code_type: str = "heavy_hex",
    experiment_type: str = EXPERIMENT_TYPE_MEMORY,
    noise_model: Optional["NoiseModel"] = None,
) -> ThresholdScenarioResult:
    """
    Run a threshold scenario sweep using either memory or CNOT experiment.
    
    Args:
        scenario: The threshold scenario to run
        study_cfg: Study configuration (shots, seed)
        progress: Optional progress callback
        code_type: Surface code layout ("heavy_hex" or "standard")
        experiment_type: Either "memory" (default) or "cnot"
        noise_model: Optional device-aware noise model. If provided, overrides
            the phenomenological p_x/p_z error rates with per-qubit noise.
            The scenario's physical_error_grid is still used to scale the noise.
        
    Returns:
        ThresholdScenarioResult with logical error rates for each (distance, p) point
    """
    if experiment_type not in (EXPERIMENT_TYPE_MEMORY, EXPERIMENT_TYPE_CNOT):
        raise ValueError(f"Unknown experiment_type: {experiment_type}. "
                        f"Must be '{EXPERIMENT_TYPE_MEMORY}' or '{EXPERIMENT_TYPE_CNOT}'.")
    
    sweeps: List[DistanceSweepResult] = []
    
    for distance in scenario.distances:
        points: List[ThresholdPoint] = []
        
        if experiment_type == EXPERIMENT_TYPE_MEMORY:
            # Memory experiment: build surface code model and use standard builder
            model = build_surface_code_model(distance, code_type=code_type)
            builder = PhenomenologicalStimBuilder(
                code=model.code,
                z_stabilizers=model.z_stabilizers,
                x_stabilizers=model.x_stabilizers,
                logical_z=model.logical_z,
                logical_x=model.logical_x,
            )
            
            for p_x, p_z in scenario.px_pz_pairs():
                stim_cfg = PhenomenologicalStimConfig(
                    rounds=scenario.rounds_for_distance(distance),
                    p_x_error=p_x,
                    p_z_error=p_z,
                    init_label=scenario.init_label,
                    family=("Z" if isinstance(scenario, XOnlyScenario) else
                            "X" if isinstance(scenario, ZOnlyScenario) else None),
                    noise_model=noise_model,
                )
                result: SimulationResult = run_logical_error_rate(
                    builder,
                    stim_cfg,
                    MonteCarloConfig(shots=study_cfg.shots, seed=study_cfg.seed),
                )

                points.append(
                    ThresholdPoint(
                        p_x=p_x,
                        p_z=p_z,
                        logical_error_rate=result.logical_error_rate,
                        avg_syndrome_weight=result.avg_syndrome_weight,
                        click_rate=result.click_rate,
                    )
                )
                if progress is not None:
                    progress(scenario, distance, p_x, p_z)
        
        else:  # CNOT experiment
            # CNOT experiment: use lattice-surgery CNOT circuit
            # The CNOT circuit tracks two observables:
            #   - Observable 0: Z on control (Z_C) - sensitive to X errors
            #   - Observable 1: X on target (X_T) - sensitive to Z errors
            # Select based on scenario.track:
            #   - track="Z" means we're tracking Z logical (use observable 0)
            #   - track="X" means we're tracking X logical (use observable 1)
            observable_idx = 0 if scenario.track == "Z" else 1
            rounds = scenario.rounds_for_distance(distance)
            
            for p_x, p_z in scenario.px_pz_pairs():
                result: SimulationResult = run_cnot_logical_error_rate(
                    distance=distance,
                    code_type=code_type,
                    p_x=p_x,
                    p_z=p_z,
                    shots=study_cfg.shots,
                    seed=study_cfg.seed,
                    rounds_pre=rounds,
                    rounds_merge=rounds,
                    rounds_post=rounds,
                    verbose=False,
                )

                # Pick the correct logical error rate based on what we're tracking
                ler = (result.logical_error_rates[observable_idx] 
                       if len(result.logical_error_rates) > observable_idx 
                       else result.logical_error_rate)

                points.append(
                    ThresholdPoint(
                        p_x=p_x,
                        p_z=p_z,
                        logical_error_rate=ler,
                        avg_syndrome_weight=result.avg_syndrome_weight,
                        click_rate=result.click_rate,
                    )
                )
                if progress is not None:
                    progress(scenario, distance, p_x, p_z)
        
        sweeps.append(DistanceSweepResult(distance=distance, points=points))
    
    return ThresholdScenarioResult(
        name=scenario.name,
        init_label=scenario.init_label,
        track=scenario.track,
        distances=list(scenario.distances),
        sweeps=sweeps,
    )


def estimate_crossings(
    result: ThresholdScenarioResult,
    min_logical_error_rate: float | None = None,
) -> Dict[tuple[int, int], float | None]:
    """Estimate threshold crossings by comparing neighbouring distances.

    Args:
        result: Threshold sweep data for one scenario
        min_logical_error_rate: Ignore sign flips that only occur while all
            neighbouring points are below this rate (default: no filtering).
            This avoids spurious “thresholds” triggered by single-count noise
            when logical error rates are effectively zero.
    """
    if len(result.sweeps) < 2:
        return {}
    crossings: Dict[tuple[int, int], float | None] = {}
    for sweep_low, sweep_high in zip(result.sweeps, result.sweeps[1:]):
        low = np.array([pt.logical_error_rate for pt in sweep_low.points])
        high = np.array([pt.logical_error_rate for pt in sweep_high.points])
        # Use the same physical x-axis convention as plotting:
        #   track == 'Z'  -> x-axis is p_x (X faults drive Z-logicals)
        #   track == 'X'  -> x-axis is p_z
        physical = np.array([pt.p_x if result.track == 'Z' else pt.p_z for pt in sweep_low.points])
        diff = low - high
        crossing_p = None
        for i in range(len(diff) - 1):
            if min_logical_error_rate is not None:
                rates = (
                    low[i],
                    high[i],
                    low[i + 1],
                    high[i + 1],
                )
                # Require all four neighbouring points to clear the noise floor
                if any(r < min_logical_error_rate for r in rates):
                    continue
            # Skip exact or near-zero differences (flat floor region)
            if diff[i] == 0 or diff[i + 1] == 0:
                continue
            # Detect a sign change between neighboring non-zero points
            if diff[i] * diff[i + 1] < 0:
                p1, p2 = physical[i], physical[i + 1]
                y1, y2 = diff[i], diff[i + 1]
                crossing_p = p1 - y1 * (p2 - p1) / (y2 - y1)
                break
        crossings[(sweep_low.distance, sweep_high.distance)] = crossing_p
    return crossings


@dataclass
class ThresholdEstimate:
    """Threshold estimate derived from pairwise crossings."""
    best_estimate: Optional[float]  # From highest distance pair
    weighted_average: Optional[float]  # Weighted by distance (higher = more weight)
    simple_average: Optional[float]  # Simple mean of all crossings
    num_crossings: int  # Number of valid crossings used
    highest_distance_pair: Optional[tuple[int, int]]  # Which distance pair gave best_estimate


def estimate_threshold(crossings: Dict[tuple[int, int], float | None]) -> ThresholdEstimate:
    """
    Estimate a single threshold value from pairwise crossing data.
    
    Uses multiple methods:
    - best_estimate: The crossing from the highest distance pair (most accurate)
    - weighted_average: Average weighted by the minimum distance in each pair
    - simple_average: Simple mean of all valid crossings
    
    Args:
        crossings: Dictionary mapping (d_low, d_high) pairs to crossing values
        
    Returns:
        ThresholdEstimate with the computed values
    """
    # Filter to valid (non-None) crossings
    valid_crossings = {k: v for k, v in crossings.items() if v is not None}
    
    if not valid_crossings:
        return ThresholdEstimate(
            best_estimate=None,
            weighted_average=None,
            simple_average=None,
            num_crossings=0,
            highest_distance_pair=None,
        )
    
    # Simple average
    values = list(valid_crossings.values())
    simple_avg = float(np.mean(values))
    
    # Weighted average (weight by minimum distance in pair)
    weights = []
    weighted_values = []
    for (d1, d2), val in valid_crossings.items():
        weight = min(d1, d2)  # Higher distances get more weight
        weights.append(weight)
        weighted_values.append(val)
    
    weights_arr = np.array(weights, dtype=float)
    weighted_values_arr = np.array(weighted_values)
    weighted_avg = float(np.average(weighted_values_arr, weights=weights_arr))
    
    # Best estimate from highest distance pair
    highest_pair = max(valid_crossings.keys(), key=lambda x: min(x))
    best_est = valid_crossings[highest_pair]
    
    return ThresholdEstimate(
        best_estimate=best_est,
        weighted_average=weighted_avg,
        simple_average=simple_avg,
        num_crossings=len(valid_crossings),
        highest_distance_pair=highest_pair,
    )


def create_standard_scenarios(distances: Sequence[int], physical_grid: Sequence[float]) -> List[ThresholdScenario]:
    scenarios: List[ThresholdScenario] = [
        XOnlyScenario(
            name="x_only",
            init_label="0",
            track="Z",
            distances=distances,
            physical_error_grid=physical_grid,
        ),
        ZOnlyScenario(
            name="z_only",
            init_label="+",
            track="X",
            distances=distances,
            physical_error_grid=physical_grid,
        ),
        SymmetricScenario(
            name="symmetric_Zinit",
            init_label="0",
            track="Z",
            distances=distances,
            physical_error_grid=physical_grid,
        ),
        SymmetricScenario(
            name="symmetric_Xinit",
            init_label="+",
            track="X",
            distances=distances,
            physical_error_grid=physical_grid,
        ),
    ]
    return scenarios
