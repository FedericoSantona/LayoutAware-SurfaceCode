"""Threshold sweep utilities for surface-code logical error studies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pymatching as pm
import stim

from surface_code import (
    PhenomenologicalStimConfig,
    build_surface_code_model,
    GlobalStimBuilder,
    MeasureRound,
)
from surface_code.dem_diagnostics import log_dem_diagnostics, env_dem_debug_enabled
from surface_code.dem_utils import (
    single_detector_hook_ids,
    circuit_to_graphlike_dem,
    add_spatial_correlations_to_dem,
    add_boundary_hooks_to_dem,
    enforce_component_boundaries,
)
from surface_code.pauli import parse_init_label
from surface_code.memory_layout import build_memory_layout


@dataclass
class MonteCarloConfig:
    shots: int = 5000
    seed: Optional[int] = None


@dataclass
class SimulationResult:
    logical_error_rate: float
    avg_syndrome_weight: float
    click_rate: float
    shots: int
    predictions: np.ndarray
    logical_observables: np.ndarray
    num_detectors: int
    observable_basis: Tuple[str, ...]
    # Optional physics-based end-basis reporting (demo readout)
    demo_bits: Optional[np.ndarray] = None
    demo_basis: Optional[str] = None


def run_logical_error_rate(
    model,
    distance: int,
    stim_config: PhenomenologicalStimConfig,
    mc_config: MonteCarloConfig,
    *,
    use_stim_memory: bool = False,
) -> SimulationResult:
    """Run logical error rate simulation for a single-patch code model.
    
    Args:
        model: Code model with attributes: code, z_stabilizers, x_stabilizers, logical_z, logical_x
        stim_config: Stim circuit configuration
        mc_config: Monte Carlo configuration
    """
    rounds = max(1, int(getattr(stim_config, "rounds", distance) or distance))
    if use_stim_memory:
        circuit, observable_pairs, metadata, bracket_map = _build_stim_memory_circuit(
            stim_config.bracket_basis or "Z",
            distance,
            rounds,
            stim_config.p_x_error,
            stim_config.p_z_error,
            stim_config.p_meas,
            stim_config.init_label,
        )
        layout = None
    else:
        layout, ops, bracket_map = build_memory_layout(
            model,
            distance,
            rounds=rounds,
            bracket_basis=stim_config.bracket_basis or "Z",
            family=stim_config.family,
        )

        builder = GlobalStimBuilder(layout)
        circuit, observable_pairs, metadata = builder.build(
            ops,
            stim_config,
            bracket_map,
            explicit_logical_start=True,
        )
        fixed_pairs: List[Tuple[int, int]] = []
        for start_idx, end_idx in observable_pairs:
            if end_idx is None:
                raise RuntimeError("Observable end index missing for memory experiment.")
            if start_idx is None:
                start_idx = end_idx
            fixed_pairs.append((start_idx, end_idx))
        observable_pairs = fixed_pairs
    
    # Validate that observables were properly created for single-patch memory experiments.
    # In multi-patch scenarios with merges, some patches may legitimately not have observables
    # due to conflicts, but for single-patch memory experiments, we must have at least one.
    if not observable_pairs:
        raise RuntimeError(
            f"No observables were created for memory experiment! "
            f"This indicates the start index was not captured. "
            f"Config: rounds={stim_config.rounds}, family={stim_config.family}, "
            f"bracket_basis={stim_config.bracket_basis}, init_label={stim_config.init_label}. "
            f"Note: This validation only applies to single-patch memory experiments."
        )

    if not use_stim_memory and len(layout.patches) == 1:
        explicit_logicals = bool(metadata.get("explicit_logical_brackets")) if metadata else False
        if not explicit_logicals:
            raise RuntimeError(
                "Expected explicit logical bracketing MPPs for single-patch memory experiment,"
                " but builder reported they were not emitted."
            )
    
    # Validate that observable pairs have both start and end indices.
    # This ensures proper tracking from initialization to final measurement.
    for i, (start_idx, end_idx) in enumerate(observable_pairs):
        if not use_stim_memory and (start_idx is None or end_idx is None):
            raise RuntimeError(
                f"Observable pair {i} has None indices: start={start_idx}, end={end_idx}. "
                f"This should not happen - observables must have both start and end indices "
                f"for proper logical error tracking in memory experiments."
            )

    return _run_circuit_logical_error_rate(circuit, observable_pairs, stim_config, mc_config, metadata)


def _run_circuit_logical_error_rate(
    circuit: stim.Circuit,
    observable_pairs: Sequence[Tuple[int, int]],
    stim_config: PhenomenologicalStimConfig,
    mc_config: MonteCarloConfig,
    metadata: Optional[dict] = None,
) -> SimulationResult:
    if metadata is None:
        metadata = {}
    dem = circuit_to_graphlike_dem(circuit)
    dem = add_spatial_correlations_to_dem(dem, metadata)
    dem = add_boundary_hooks_to_dem(dem, metadata)
    boundary_meta = (metadata.get("boundary_anchors", {}) or {}).get("detector_ids")
    enforce_component_boundaries(dem, explicit_anchor_ids=boundary_meta)
    dem_debug = env_dem_debug_enabled()
    log_dem_diagnostics("memory-pre", dem, metadata, enabled=dem_debug)

    metadata.setdefault("boundary_anchors", {})["detector_ids"] = list(single_detector_hook_ids(dem))
    log_dem_diagnostics("memory-post", dem, metadata, enabled=dem_debug)

    # Build matcher on the raw DEM first. On well-formed memory experiments
    # (closed space-time cycles), components naturally have even parity and
    # decoding is feasible without synthetic boundaries.
    matcher = pm.Matching.from_detector_error_model(dem)

    # Sample detector and observable data
    dem_sampler = dem.compile_sampler(seed=mc_config.seed)
    detector_samples, observable_samples, _ = dem_sampler.sample(mc_config.shots)
    detector_samples_bool = np.asarray(detector_samples, dtype=np.bool_)
    
    # Validate that observables were sampled correctly
    if observable_samples is None or observable_samples.size == 0:
        raise RuntimeError(
            f"No observable samples were generated! "
            f"This indicates that no OBSERVABLE_INCLUDE operations were emitted in the circuit. "
            f"Observable pairs count: {len(observable_pairs)}"
        )
    
    logical_array = np.asarray(observable_samples, dtype=np.uint8)
    
    # Ensure we have the expected number of observables
    if logical_array.shape[1] != len(observable_pairs):
        raise RuntimeError(
            f"Mismatch between observable pairs ({len(observable_pairs)}) "
            f"and sampled observables ({logical_array.shape[1]})"
        )

    # Decode
    predictions = matcher.decode_batch(detector_samples_bool)
    predictions = np.asarray(predictions, dtype=np.uint8)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    detector_samples_uint8 = detector_samples_bool.astype(np.uint8)
    target_bits = logical_array[:, 0]
    logical_error_rate = (predictions[:, 0] ^ target_bits).mean()

    avg_syndrome_weight = detector_samples_uint8.sum(axis=1).mean()
    click_rate = (detector_samples_uint8.sum(axis=1) > 0).mean()

    # Prefer basis labels provided by builder metadata, fallback to inference
    if metadata is not None and "observable_basis" in metadata:
        basis_tuple = tuple(str(b) for b in metadata.get("observable_basis", tuple()))
        if len(basis_tuple) != predictions.shape[1]:
            observable_basis = _infer_observable_basis(predictions.shape[1], stim_config)
        else:
            observable_basis = basis_tuple
    else:
        observable_basis = _infer_observable_basis(predictions.shape[1], stim_config)

    # Optionally sample demo measurement bits from the circuit's measurement
    # record (end-only MPP in requested basis). This is independent from DEM.
    demo_bits: Optional[np.ndarray] = None
    demo_basis: Optional[str] = None
    if metadata is not None and "demo_index" in metadata:
        demo_basis = metadata.get("demo_basis")
        # Compile a circuit sampler to sample raw measurements including the demo.
        circ_sampler = circuit.compile_sampler(seed=mc_config.seed)
        # Sample measurements only (detector samples not needed here).
        m_samples = circ_sampler.sample(shots=mc_config.shots)
        # The returned array is [shots, num_measurements]; pick the column at demo_index.
        # Stim returns booleans; convert to uint8 for consistency.
        demo_col = int(metadata["demo_index"]) if metadata["demo_index"] is not None else None
        if demo_col is not None:
            demo_bits = np.asarray(m_samples[:, demo_col], dtype=np.uint8)

    return SimulationResult(
        logical_error_rate=float(logical_error_rate),
        avg_syndrome_weight=float(avg_syndrome_weight),
        click_rate=float(click_rate),
        shots=mc_config.shots,
        predictions=predictions,
        logical_observables=logical_array,
        num_detectors=dem.num_detectors,
        observable_basis=observable_basis,
        demo_bits=demo_bits,
        demo_basis=demo_basis,
    )


def _infer_observable_basis(num_observables: int, stim_config: PhenomenologicalStimConfig) -> Tuple[str, ...]:
    """Infer logical basis labels for each tracked observable column."""
    if num_observables == 0:
        return tuple()
    # Bracket basis is fixed for DEM/decoding
    if stim_config.bracket_basis is not None:
        basis = stim_config.bracket_basis.strip().upper()
    elif stim_config.init_label is not None:
        basis, _ = parse_init_label(stim_config.init_label)
    else:
        basis = "Z"
    if basis not in {"X", "Z"}:
        raise ValueError(f"Unable to infer logical basis for observables (got '{basis}')")
    return tuple(basis for _ in range(num_observables))


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
        # For surface codes, we need at least 'distance' rounds to allow proper error correction
        # Using distance ensures sufficient time for error correction to work
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
    use_stim_memory: bool = False


def _build_stim_memory_circuit(
    track: str,
    distance: int,
    rounds: int,
    p_x: float,
    p_z: float,
    p_meas: float,
    init_label: str | None,
) -> tuple[stim.Circuit, List[Tuple[int, int]], Dict[str, object], Dict[str, str]]:
    track_u = (track or "Z").strip().upper()
    task = "surface_code:rotated_memory_z" if track_u == "Z" else "surface_code:rotated_memory_x"
    data_prob = float(max(p_x, p_z))
    circuit = stim.Circuit.generated(
        task,
        distance=int(distance),
        rounds=int(rounds),
        before_round_data_depolarization=data_prob,
        before_measure_flip_probability=float(p_meas),
        after_reset_flip_probability=float(p_meas),
    )
    metadata: Dict[str, object] = {
        "observable_basis": (track_u,),
        "observable_patches": ("q0",),
        "boundary_anchors": {"detector_ids": []},
        "mwpm_debug": {},
        "explicit_logical_brackets": False,
        "stim_generator": {
            "task": task,
            "distance": distance,
            "rounds": rounds,
        },
        "noise_model": {
            "p_x_error": float(p_x),
            "p_z_error": float(p_z),
            "p_meas": float(p_meas),
        },
    }
    observable_pairs = [(0, 0)]
    bracket_map = {"q0": track_u}
    return circuit, observable_pairs, metadata, bracket_map


def run_scenario(
    scenario: ThresholdScenario,
    study_cfg: ThresholdStudyConfig,
    code_type: str = "heavy_hex",
    progress: Callable[[ThresholdScenario, int, float, float], None] | None = None,
    p_meas_override: float | None = None,
    p_meas_scale: float = 1.0,
) -> ThresholdScenarioResult:
    sweeps: List[DistanceSweepResult] = []
    for distance in scenario.distances:
        model = build_surface_code_model(distance, code_type=code_type)
        points: List[ThresholdPoint] = []
        for p_x, p_z in scenario.px_pz_pairs():
            # Calculate p_meas using unambiguous auto rule
            if p_meas_override is not None:
                p_meas = p_meas_override
            else:
                # Auto rule: p_meas = p for symmetric cases, p_meas = max(p_x, p_z) for asymmetric cases
                if isinstance(scenario, SymmetricScenario):
                    p_meas = p_x  # p_x == p_z for symmetric scenarios
                else:
                    p_meas = max(p_x, p_z)  # For X-only or Z-only scenarios
                p_meas *= p_meas_scale
            
            # Clamp p_meas to valid range [0.0, 0.499999]
            p_meas = min(max(p_meas, 0.0), 0.499999)
            
            stim_cfg = PhenomenologicalStimConfig(
                rounds=scenario.rounds_for_distance(distance),
                p_x_error=p_x,
                p_z_error=p_z,
                family=("Z" if isinstance(scenario, XOnlyScenario) else
                        "X" if isinstance(scenario, ZOnlyScenario) else None),
                bracket_basis=scenario.track,
                init_label=scenario.init_label,
                p_meas=p_meas,
            )
            result: SimulationResult = run_logical_error_rate(
                model,
                distance,
                stim_cfg,
                MonteCarloConfig(shots=study_cfg.shots, seed=study_cfg.seed),
                use_stim_memory=study_cfg.use_stim_memory,
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


def create_standard_scenarios(
    distances: Sequence[int],
    physical_grid: Sequence[float],
    rounds_scale: float = 1.0,
    code_type: str = "heavy_hex",
) -> List[ThresholdScenario]:
    scenarios: List[ThresholdScenario] = [
        XOnlyScenario(
            name="x_only",
            init_label="0",
            track="Z",
            distances=distances,
            physical_error_grid=physical_grid,
            rounds_scale=rounds_scale,
        ),
        ZOnlyScenario(
            name="z_only",
            init_label="+",
            track="X",
            distances=distances,
            physical_error_grid=physical_grid,
            rounds_scale=rounds_scale,
        ),
        SymmetricScenario(
            name="symmetric_Zinit",
            init_label="0",
            track="Z",
            distances=distances,
            physical_error_grid=physical_grid,
            rounds_scale=rounds_scale,
        ),
        SymmetricScenario(
            name="symmetric_Xinit",
            init_label="+",
            track="X",
            distances=distances,
            physical_error_grid=physical_grid,
            rounds_scale=rounds_scale,
        ),
    ]
    return scenarios
