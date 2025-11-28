"""Monte Carlo simulation routines for logical error estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from collections import Counter

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
    logical_error_rates: List[float]
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
    verbose: bool = False,
) -> SimulationResult:
    circuit, observable_pairs = builder.build(stim_config)
    return run_circuit_logical_error_rate(circuit, observable_pairs, stim_config, mc_config, verbose=verbose)


def run_circuit_logical_error_rate(
    circuit: stim.Circuit,
    observable_pairs: Sequence[Tuple[int, int]],
    stim_config: PhenomenologicalStimConfig,
    mc_config: MonteCarloConfig,
    verbose: bool = False
) -> SimulationResult:

    
    # 1. Get DEM WITHOUT decomposition (before)
    print("\n=== Checking for high-weight errors ===")
    dem_before = circuit.detector_error_model(decompose_errors=False)
    counts_before = Counter(
        sum(1 for t in inst.targets_copy() if t.is_relative_detector_id())
        for inst in dem_before
        if inst.type == 'error'
    )
    print(f"Detectors: {dem_before.num_detectors}")
    print(f"Total error instructions: {sum(counts_before.values())}")
    print(f"Error size histogram (before): {dict(sorted(counts_before.items()))}")
    
    # 2. Check if there are high-weight errors that need decomposition
    high_weight_count = 0
    for inst in dem_before:
        if inst.type == 'error':
            num_detectors = sum(1 for t in inst.targets_copy() if t.is_relative_detector_id())
            if num_detectors > 2:  # These need decomposition
                high_weight_count += 1

    if high_weight_count == 0:
        print("No high-weight errors found. Using undecomposed DEM.")
        dem = dem_before

    elif high_weight_count > 0:
        # 3. Try decomposition WITHOUT ignoring failures to see what fails
        print(f"\n=== CHECKING FOR DECOMPOSITION FAILURES ({high_weight_count} high-weight errors found) ===")
        try:
            dem_strict = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=False)
            print("✓ All errors decomposed successfully!")
            # Use decomposed DEM
            dem = dem_strict
        except Exception as e:
            print(f"✗ Decomposition failed: {e}")
            print("\nTo investigate further, examine the circuit's error model structure.")
            # You can optionally write the DEM to file for manual inspection
            try:
                with open("dem_before_decomposition.txt", "w") as f:
                    f.write(str(dem_before))
                print("Wrote undecomposed DEM to 'dem_before_decomposition.txt'")
            except Exception:
                pass  # Don't fail if file write fails
            
            # Get DEM WITH decomposition, ignoring failures (after)
            print("\n=== AFTER DECOMPOSITION (ignoring failures) ===")
            print("We are decomposing errors while ignoring failures")
            dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
            counts_after = Counter(
                sum(1 for t in inst.targets_copy() if t.is_relative_detector_id())
                for inst in dem
                if inst.type == 'error'
            )
            print(f"Detectors: {dem.num_detectors}")
            print(f"Total error instructions: {sum(counts_after.values())}")
            print(f"Error size histogram (after): {dict(sorted(counts_after.items()))}")
            
            # 4. Compare the results (only if decomposition failed)
            print("\n=== COMPARISON ===")
            total_before = sum(counts_before.values())
            total_after = sum(counts_after.values())
            print(f"Error instructions lost: {total_before - total_after}")
            
            # Show changes in distribution
            all_sizes = set(counts_before.keys()) | set(counts_after.keys())
            for size in sorted(all_sizes):
                before = counts_before.get(size, 0)
                after = counts_after.get(size, 0)
                change = after - before
                if change != 0:
                    print(f"  Size {size}: {before} → {after} (Δ{change:+d})")
            
            # Identify high-weight errors that might fail decomposition
            print("\n=== HIGH-WEIGHT ERRORS (potential decomposition issues) ===")
            shown_count = 0
            for i, inst in enumerate(dem_before):
                if inst.type == 'error':
                    num_detectors = sum(1 for t in inst.targets_copy() if t.is_relative_detector_id())
                    if num_detectors > 2:  # These need decomposition
                        shown_count += 1
                        if shown_count <= 5:  # Show first 5 as examples
                            print(f"Error {i}: weight={num_detectors}, probability={inst.args_copy()[0]}")
                            print(f"  Targets: {inst.targets_copy()[:10]}...")  # Show first 10 targets
            if shown_count > 5:
                print(f"... and {shown_count - 5} more high-weight errors")
    else:
        print("\n=== NO DECOMPOSITION NEEDED ===")
        print("No high-weight errors (>2 detectors) found. Using undecomposed DEM.")
        dem = dem_before
    
    print("\ndetectors:", dem.num_detectors)
    matcher = pm.Matching.from_detector_error_model(dem)

    print("The matcher object is:",matcher)               # will say “… X detectors, Y boundary node(s), Z edges”
    print("Boundary nodes:", matcher.boundary)

    dem_sampler = dem.compile_sampler(seed=mc_config.seed)
    detector_samples, observable_samples, _ = dem_sampler.sample(mc_config.shots)

    detector_samples_bool = np.asarray(detector_samples, dtype=np.bool_)
    if observable_samples is None or observable_samples.size == 0:
        logical_array = np.zeros((mc_config.shots, 1), dtype=np.uint8)
    else:
        logical_array = np.asarray(observable_samples, dtype=np.uint8)

    # Decode with error handling
    try:
        predictions = matcher.decode_batch(detector_samples_bool)
    except ValueError as e:
        error_msg = str(e)
        if "No perfect matching could be found" in error_msg:
            # This error typically means the DEM lacks boundary nodes for some components
            # Try to provide a helpful error message
            raise RuntimeError(
                f"PyMatching decoding failed: {error_msg}. "
                "This may indicate that the detector error model lacks boundary nodes "
                "for some connected components. Check that the circuit properly defines "
                "boundary detectors or use a code type that supports boundary handling."
            ) from e
        else:
            raise
    
    predictions = np.asarray(predictions, dtype=np.uint8)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    detector_samples_uint8 = detector_samples_bool.astype(np.uint8)
    
    # Compute error rates for all observables
    num_observables = logical_array.shape[1] if logical_array.ndim > 1 else 1
    logical_error_rates: List[float] = []
    
    # Ensure predictions has enough columns
    if predictions.shape[1] < num_observables:
        # Pad predictions with zeros if needed
        padding = np.zeros((predictions.shape[0], num_observables - predictions.shape[1]), dtype=np.uint8)
        predictions = np.hstack([predictions, padding])
    
    for i in range(num_observables):
        if i < logical_array.shape[1] and i < predictions.shape[1]:
            ler = (predictions[:, i] ^ logical_array[:, i]).mean()
            logical_error_rates.append(float(ler))
        else:
            # Fallback: if observable doesn't exist, error rate is 0
            logical_error_rates.append(0.0)
    
    # Backward compatibility: logical_error_rate is the first observable's error rate
    logical_error_rate = logical_error_rates[0] if logical_error_rates else 0.0

    avg_syndrome_weight = detector_samples_uint8.sum(axis=1).mean()
    click_rate = (detector_samples_uint8.sum(axis=1) > 0).mean()

    return SimulationResult(
        logical_error_rate=float(logical_error_rate),
        logical_error_rates=logical_error_rates,
        avg_syndrome_weight=float(avg_syndrome_weight),
        click_rate=float(click_rate),
        shots=mc_config.shots,
        predictions=predictions,
        logical_observables=logical_array,
        num_detectors=dem.num_detectors,
    )


