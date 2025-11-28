# src/simulation/physics_runner.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional

import numpy as np
import stim
from .montecarlo import MonteCarloConfig


@dataclass
class PhysicsResult:
    shots: int
    correlators: Dict[str, float]
    raw_samples: Optional[np.ndarray] = None


def run_circuit_physics(
    circuit: stim.Circuit,
    correlator_map: Dict[str, Sequence[int]],
    mc_config: MonteCarloConfig,
    keep_samples: bool = False,
    verbose: bool = False,
) -> PhysicsResult:
    """Estimate correlators by direct sampling of a Stim circuit.

    This is the "physics mode" runner: it does *not* build a detector error
    model or use PyMatching. It simply:

      1. Compiles a Stim sampler for the given circuit.
      2. Samples all measurement outcomes for `mc_config.shots` shots.
      3. For each requested correlator, multiplies the ±1 outcomes of the
         specified measurements and averages over shots.

    Parameters
    ----------
    circuit:
        Full Stim circuit, including any stabilizer rounds and the final MPP
        measurements whose indices you want to use as correlators.
    correlator_map:
        Dict mapping a human-readable name to a list of measurement indices.

        Each value is a sequence of absolute measurement indices in the
        Stim circuit (0-based, in the order Stim records measurements).
        For example:

            {
                "XX": [idx_xx],                          # Bell XX stabilizer
                "ZZ": [idx_zz],                          # Bell ZZ stabilizer
                "XXX_GHZ": [idx_X1, idx_X2, idx_X3],    # GHZ stabilizer
            }

        These indices are exactly the ones returned by the
        `PhenomenologicalStimBuilder.measure_logical_once` calls.
    mc_config:
        Monte Carlo configuration (shots, seed).
    keep_samples:
        If True, include the raw measurement bit array in the result
        (useful for debugging / additional analysis).
    verbose:
        If True, print correlator estimates to stdout.

    Returns
    -------
    PhysicsResult
        Contains the estimated correlators and (optionally) the raw samples.
    """

    # 1. Compile sampler directly from the circuit (no DEM, no matching).
    sampler = circuit.compile_sampler(seed=mc_config.seed)

    # 2. Sample all measurements: shape (shots, num_measurements)
    samples = sampler.sample(mc_config.shots)
    samples = np.asarray(samples, dtype=np.float64)

    # 3. Map bits {0,1} to ±1 using convention 0 -> +1, 1 -> -1
    #    This matches the usual Pauli measurement sign convention.
    pm1 = 1 - 2 * samples  # same shape as `samples`, entries in {+1, -1}

    correlators: Dict[str, float] = {}

    for name, idxs in correlator_map.items():
        # Allow a single index or a sequence
        if isinstance(idxs, (int, np.integer)):
            idx_list = [int(idxs)]
        else:
            idx_list = [int(i) for i in idxs]

        if len(idx_list) == 0:
            # Degenerate case: no indices => correlator is trivially +1
            correlators[name] = 1.0
            if verbose:
                print(f"{name}: no indices provided, setting ⟨{name}⟩ = 1.0")
            continue

        # 4. Product of ±1 outcomes across the listed measurements, per shot
        #    pm1[:, idx_list] has shape (shots, len(idx_list))
        vals = pm1[:, idx_list].prod(axis=1)  # shape (shots,)

        # 5. Average over shots gives the estimated expectation value
        correlators[name] = float(vals.mean())

        if verbose:
            print(f"{name}: ⟨{name}⟩ ≈ {correlators[name]:.6f}")

    return PhysicsResult(
        shots=mc_config.shots,
        correlators=correlators,
        raw_samples=samples if keep_samples else None,
    )