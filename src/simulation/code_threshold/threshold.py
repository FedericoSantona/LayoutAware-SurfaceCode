"""Threshold sweep utilities for surface-code logical error studies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

from surface_code import (
    PhenomenologicalStimBuilder,
    PhenomenologicalStimConfig,
    build_surface_code_model,
)

from ..ler_simulator import MonteCarloConfig, SimulationResult, run_logical_error_rate


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
) -> ThresholdScenarioResult:
    sweeps: List[DistanceSweepResult] = []
    for distance in scenario.distances:
        model = build_surface_code_model(distance, code_type=code_type)
        builder = PhenomenologicalStimBuilder(
            code=model.code,
            z_stabilizers=model.z_stabilizers,
            x_stabilizers=model.x_stabilizers,
            logical_z=model.logical_z,
            logical_x=model.logical_x,
        )
        points: List[ThresholdPoint] = []
        for p_x, p_z in scenario.px_pz_pairs():
            stim_cfg = PhenomenologicalStimConfig(
                rounds=scenario.rounds_for_distance(distance),
                p_x_error=p_x,
                p_z_error=p_z,
                init_label=scenario.init_label,
                family=("Z" if isinstance(scenario, XOnlyScenario) else
                        "X" if isinstance(scenario, ZOnlyScenario) else None)
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
        sweeps.append(DistanceSweepResult(distance=distance, points=points))
    return ThresholdScenarioResult(
        name=scenario.name,
        init_label=scenario.init_label,
        track=scenario.track,
        distances=list(scenario.distances),
        sweeps=sweeps,
    )


def estimate_crossings(result: ThresholdScenarioResult) -> Dict[tuple[int, int], float | None]:
    """Estimate threshold crossings by comparing neighbouring distances."""
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
