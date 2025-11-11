#!/usr/bin/env python3
"""Test script to verify error suppression with distance for Bell vs Single Patch."""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from surface_code import build_surface_code_model, PhenomenologicalStimConfig
from surface_code.memory_layout import build_memory_layout
from surface_code.builder import GlobalStimBuilder
from surface_code.surgery_compile import compile_circuit_to_surgery
from surface_code.layout import PatchObject
from surface_code.dem_utils import circuit_to_graphlike_dem
from experiments.code_threshold.threshold import _run_circuit_logical_error_rate, MonteCarloConfig
from qiskit import QuantumCircuit
import numpy as np

def test_single_patch(distance, shots=10000):
    """Test single patch error suppression."""
    print(f"\n=== Testing Single Patch (distance={distance}) ===")
    
    model = build_surface_code_model(distance, code_type="standard")
    layout, ops, bracket_map = build_memory_layout(
        model,
        distance,
        rounds=distance,
        bracket_basis="Z",
        family=None,
    )
    
    stim_cfg = PhenomenologicalStimConfig(
        rounds=distance,
        p_x_error=5e-3,
        p_z_error=5e-3,
        p_meas=5e-3,
        init_label="0",
        bracket_basis="Z",
    )
    
    builder = GlobalStimBuilder(layout)
    circuit, observable_pairs, metadata = builder.build(
        ops,
        stim_cfg,
        bracket_map,
        explicit_logical_start=True,
    )
    
    # Fix observable pairs
    fixed_pairs = []
    for start_idx, end_idx in observable_pairs:
        if end_idx is None:
            raise RuntimeError("Observable end index missing")
        if start_idx is None:
            start_idx = end_idx
        fixed_pairs.append((start_idx, end_idx))
    
    result = _run_circuit_logical_error_rate(
        circuit,
        fixed_pairs,
        stim_cfg,
        MonteCarloConfig(shots=shots, seed=46),
        metadata,
    )
    
    ler = result.logical_error_rate
    print(f"  Logical Error Rate: {ler:.6f}")
    return ler

def test_bell_state(distance, shots=10000):
    """Test Bell state error suppression."""
    print(f"\n=== Testing Bell State (distance={distance}) ===")
    
    # Build Bell circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    model = build_surface_code_model(distance, code_type="standard")
    template_patch = PatchObject.from_code_model(model)
    
    # Create patches
    patches = {}
    for i in range(2):
        patch = template_patch.with_offset(0, i * 5.0, 0.0)
        patches[f"q{i}"] = patch
    
    seams = {}
    bracket_map = {"q0": "Z", "q1": "Z"}
    
    layout, ops = compile_circuit_to_surgery(
        qc, patches, seams, distance=distance,
        bracket_map=bracket_map, warmup_rounds=1
    )
    
    stim_cfg = PhenomenologicalStimConfig(
        rounds=distance,
        p_x_error=5e-3,
        p_z_error=5e-3,
        p_meas=5e-3,
        init_label="0",
        bracket_basis="Z",
    )
    
    builder = GlobalStimBuilder(layout)
    circuit, observable_pairs, metadata = builder.build(
        ops,
        stim_cfg,
        bracket_map,
        qc,
    )
    
    result = _run_circuit_logical_error_rate(
        circuit,
        observable_pairs,
        stim_cfg,
        MonteCarloConfig(shots=shots, seed=46),
        metadata,
    )
    
    # Average LER across both qubits
    ler = result.logical_error_rate
    print(f"  Logical Error Rate: {ler:.6f}")
    return ler

if __name__ == "__main__":
    distances = [3, 5, 7]
    shots = 50000
    
    print("=" * 60)
    print("ERROR SUPPRESSION TEST")
    print("=" * 60)
    
    # Test single patch
    print("\n" + "=" * 60)
    print("SINGLE PATCH STATE")
    print("=" * 60)
    single_patch_lers = {}
    for d in distances:
        ler = test_single_patch(d, shots)
        single_patch_lers[d] = ler
    
    # Test Bell state
    print("\n" + "=" * 60)
    print("BELL STATE")
    print("=" * 60)
    bell_lers = {}
    for d in distances:
        ler = test_bell_state(d, shots)
        bell_lers[d] = ler
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nSingle Patch LERs:")
    for d in distances:
        print(f"  d={d}: {single_patch_lers[d]:.6f}")
    
    print("\nBell State LERs:")
    for d in distances:
        print(f"  d={d}: {bell_lers[d]:.6f}")
    
    # Check if error suppression works
    print("\n" + "=" * 60)
    print("ERROR SUPPRESSION ANALYSIS")
    print("=" * 60)
    
    def check_suppression(name, lers):
        """Check if higher distance suppresses error."""
        ratios = []
        for i in range(len(distances) - 1):
            d1, d2 = distances[i], distances[i+1]
            ratio = lers[d1] / lers[d2] if lers[d2] > 0 else float('inf')
            ratios.append(ratio)
            print(f"  {name} d={d1}→d={d2}: LER ratio = {ratio:.3f} ({'✓' if ratio > 1.0 else '✗'})")
        return all(r > 1.0 for r in ratios)
    
    single_suppresses = check_suppression("Single Patch", single_patch_lers)
    bell_suppresses = check_suppression("Bell State", bell_lers)
    
    print(f"\nSingle Patch suppresses error with distance: {'YES ✓' if single_suppresses else 'NO ✗'}")
    print(f"Bell State suppresses error with distance: {'YES ✓' if bell_suppresses else 'NO ✗'}")
    
    if not single_suppresses and bell_suppresses:
        print("\n⚠️  ISSUE CONFIRMED: Single patch does NOT suppress error with distance, but Bell state does!")


