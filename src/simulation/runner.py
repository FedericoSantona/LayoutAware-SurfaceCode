"""Monte Carlo simulation routines for logical error estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pymatching as pm
import stim

from surface_code.stim_builder import PhenomenologicalStimBuilder, PhenomenologicalStimConfig


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


def run_logical_error_rate(
    builder: PhenomenologicalStimBuilder,
    stim_config: PhenomenologicalStimConfig,
    mc_config: MonteCarloConfig,
) -> SimulationResult:
    circuit, observable_pairs = builder.build(stim_config)
    return run_circuit_logical_error_rate(circuit, observable_pairs, stim_config, mc_config)


def run_circuit_logical_error_rate(
    circuit: stim.Circuit,
    observable_pairs: Sequence[Tuple[int, int]],
    stim_config: PhenomenologicalStimConfig,
    mc_config: MonteCarloConfig,
) -> SimulationResult:
    dem = circuit.detector_error_model()
    matcher = pm.Matching.from_detector_error_model(dem)

    dem_sampler = dem.compile_sampler(seed=mc_config.seed)
    detector_samples, observable_samples, _ = dem_sampler.sample(mc_config.shots)

    detector_samples_bool = np.asarray(detector_samples, dtype=np.bool_)
    if observable_samples is None or observable_samples.size == 0:
        logical_array = np.zeros((mc_config.shots, 1), dtype=np.uint8)
    else:
        logical_array = np.asarray(observable_samples, dtype=np.uint8)

    predictions = matcher.decode_batch(detector_samples_bool)
    predictions = np.asarray(predictions, dtype=np.uint8)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    detector_samples_uint8 = detector_samples_bool.astype(np.uint8)
    logical_error_rate = (predictions[:, 0] ^ logical_array[:, 0]).mean()

    avg_syndrome_weight = detector_samples_uint8.sum(axis=1).mean()
    click_rate = (detector_samples_uint8.sum(axis=1) > 0).mean()

    return SimulationResult(
        logical_error_rate=float(logical_error_rate),
        avg_syndrome_weight=float(avg_syndrome_weight),
        click_rate=float(click_rate),
        shots=mc_config.shots,
        predictions=predictions,
        logical_observables=logical_array,
        num_detectors=dem.num_detectors,
    )


