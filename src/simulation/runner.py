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

    det_sampler = circuit.compile_detector_sampler(seed=mc_config.seed)
    detector_samples = det_sampler.sample(mc_config.shots)

    if observable_pairs:
        sampler = circuit.compile_sampler(seed=mc_config.seed)
        meas_samples = sampler.sample(mc_config.shots)
        logical_measurements = []
        for start, end in observable_pairs:
            logical_measurements.append(meas_samples[:, start] ^ meas_samples[:, end])
        logical_array = np.vstack(logical_measurements).T.astype(np.uint8)
    else:
        logical_array = np.zeros((mc_config.shots, 1), dtype=np.uint8)

    predictions = matcher.decode_batch(detector_samples)
    predictions = np.asarray(predictions, dtype=np.uint8)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    logical_error_rate = (predictions[:, 0] ^ logical_array[:, 0]).mean()

    avg_syndrome_weight = detector_samples.sum(axis=1).mean()
    click_rate = (detector_samples.sum(axis=1) > 0).mean()

    return SimulationResult(
        logical_error_rate=float(logical_error_rate),
        avg_syndrome_weight=float(avg_syndrome_weight),
        click_rate=float(click_rate),
        shots=mc_config.shots,
        predictions=predictions,
        logical_observables=logical_array,
        num_detectors=dem.num_detectors,
    )


