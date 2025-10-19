
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, TYPE_CHECKING, Tuple

import matplotlib.pyplot as plt
import numpy as np
from qiskit_qec.operators.pauli_list import PauliList

if TYPE_CHECKING:  # pragma: no cover - import guard for type checking only
    from simulation.runner import SimulationResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def wilson_rate_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for a Bernoulli rate.
    
    Args:
        k: Number of successes
        n: Total number of trials
        z: Z-score for desired confidence level (default 1.96 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    if n == 0:
        return (0.0, 1.0)
    
    p = k / n
    z_squared = z * z
    n_plus_z_squared = n + z_squared
    
    # Wilson score interval formula
    center = (k + z_squared / 2) / n_plus_z_squared
    margin = z * np.sqrt((p * (1 - p) + z_squared / (4 * n)) / n_plus_z_squared)
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    
    return (lower, upper)


def compute_two_qubit_correlations(
    demo_z_bits: Dict[str, np.ndarray], 
    demo_x_bits: Dict[str, np.ndarray], 
    pairs: list[Tuple[str, str]], 
    shots: int
) -> Dict[str, Dict[str, float]]:
    """Compute two-qubit correlations and Bell state fidelity bounds.
    
    Args:
        demo_z_bits: Dict mapping qubit names to Z-basis demo measurement arrays
        demo_x_bits: Dict mapping qubit names to X-basis demo measurement arrays  
        pairs: List of (qubit1, qubit2) tuples to compute correlations for
        shots: Total number of shots
        
    Returns:
        Dict with correlation data for each pair:
        {
            "q0,q1": {
                "zz_correlator": float,  # ⟨Z⊗Z⟩
                "xx_correlator": float,  # ⟨X⊗X⟩
                "zz_ci": (lower, upper), # Wilson CI for ZZ parity-0 rate
                "xx_ci": (lower, upper), # Wilson CI for XX parity-0 rate
                "fidelity_bound": float  # F ≥ 0.5(⟨Z⊗Z⟩ + ⟨X⊗X⟩)
            }
        }
    """
    correlations = {}
    
    for q1, q2 in pairs:
        if q1 not in demo_z_bits or q2 not in demo_z_bits:
            continue
        if q1 not in demo_x_bits or q2 not in demo_x_bits:
            continue
            
        # Compute Z⊗Z parity: z_parity = z_demo[q1] ⊕ z_demo[q2]
        z_parity = np.bitwise_xor(demo_z_bits[q1], demo_z_bits[q2])
        z_parity_0_count = int(np.sum(z_parity == 0))
        z_parity_0_rate = z_parity_0_count / shots
        zz_correlator = 1.0 - 2.0 * (1.0 - z_parity_0_rate)  # ⟨Z⊗Z⟩ = 1 - 2·P(z_parity=1)
        zz_ci = wilson_rate_ci(z_parity_0_count, shots)
        
        # Compute X⊗X parity: x_parity = x_demo[q1] ⊕ x_demo[q2]  
        x_parity = np.bitwise_xor(demo_x_bits[q1], demo_x_bits[q2])
        x_parity_0_count = int(np.sum(x_parity == 0))
        x_parity_0_rate = x_parity_0_count / shots
        xx_correlator = 1.0 - 2.0 * (1.0 - x_parity_0_rate)  # ⟨X⊗X⟩ = 1 - 2·P(x_parity=1)
        xx_ci = wilson_rate_ci(x_parity_0_count, shots)
        
        # Bell state fidelity bound: F ≥ 0.5(⟨Z⊗Z⟩ + ⟨X⊗X⟩)
        fidelity_bound = 0.5 * (zz_correlator + xx_correlator)
        
        pair_key = f"{q1},{q2}"
        correlations[pair_key] = {
            "zz_correlator": zz_correlator,
            "xx_correlator": xx_correlator,
            "zz_ci": zz_ci,
            "xx_ci": xx_ci,
            "fidelity_bound": fidelity_bound
        }
    
    return correlations


def plot_heavy_hex_code(model, distance):
    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig = model.code.draw(
        face_colors=False,
        xcolor="lightcoral",
        zcolor="skyblue",
        figsize=(5, 5),
    )
    plt.savefig(plot_dir / f"heavy_hex_d{distance}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def diagnostic_print(model, args):

    print(f"Heavy-hex code with d={args.distance} has {model.code.n} physical qubits.")
    print(f"Number of gauge generators: {len(model.generators)}")
    print("Gauge Generators:")
    for i, gen in enumerate(model.generators):
        print(f"  GG{i}: {gen}")

    stabilizers = PauliList(model.stabilizer_matrix)
    print("Stabilizers (basis):")
    for i, stab in enumerate(stabilizers):
        print(f"  SG{i}: {stab}")

    print(
        "Using CSS-projected stabilizers: "
        f"{len(model.z_stabilizers)} Z, {len(model.x_stabilizers)} X "
        f"(rank(S)={model.stabilizer_matrix.shape[0]}, total={len(model.z_stabilizers)+len(model.x_stabilizers)})"
    )

    print("Chosen Z_L (geometry):", model.logical_z)
    print("Chosen X_L (algebraic from Z_L):", model.logical_x)

    diagnostics = model.diagnostics()
    print("Logical operator checker:")
    print(f"  Z_L commutes with all stabilizers: {diagnostics['commute_Z']}")
    print(f"  X_L commutes with all stabilizers: {diagnostics['commute_X']}")
    print(f"  Z_L and X_L anticommute: {diagnostics['anticommute']}")
    print(f"  Z_L in stabilizer group: {diagnostics['Z_in_stabilizer']}")
    print(f"  X_L in stabilizer group: {diagnostics['X_in_stabilizer']}")
    print(f"  Z_L weight: {diagnostics['weight_Z']}")
    print(f"  X_L weight: {diagnostics['weight_X']}")

    print(f"n={diagnostics['n']}, s={diagnostics['s']}, r={diagnostics['r']}  =>  k={diagnostics['k']}")




def compute_joint_correlations(joint_demo_bits: Dict[str, Dict], shots: int) -> Dict[str, Dict[str, float]]:
    """Compute two-qubit correlations from joint demo measurements.
    
    Args:
        joint_demo_bits: Dictionary mapping joint demo keys to measurement data
        shots: Number of shots for confidence interval calculation
        
    Returns:
        Dictionary mapping pair keys to correlation data
    """
    correlations = {}
    
    # Group joint demos by pair
    pair_demos = {}
    for joint_key, demo_data in joint_demo_bits.items():
        pair = demo_data["pair"]
        basis = demo_data["basis"]
        pair_key = f"{pair[0]},{pair[1]}"
        
        if pair_key not in pair_demos:
            pair_demos[pair_key] = {}
        pair_demos[pair_key][basis] = demo_data
    
    # Compute correlations for each pair
    for pair_key, demos in pair_demos.items():
        zz_data = demos.get("Z")
        xx_data = demos.get("X")
        
        if zz_data and xx_data:
            # Compute expectation values: ⟨O⟩ = 1 - 2·P(meas=1)
            zz_bits = zz_data["bits"]
            xx_bits = xx_data["bits"]
            
            zz_p1 = float(zz_bits.mean())
            xx_p1 = float(xx_bits.mean())
            
            zz_expectation = 1.0 - 2.0 * zz_p1
            xx_expectation = 1.0 - 2.0 * xx_p1
            
            # Compute Wilson CI on expectation value
            # For expectation E = 1 - 2p, we need CI on p first, then transform
            zz_p1_count = int(np.sum(zz_bits))
            xx_p1_count = int(np.sum(xx_bits))
            
            zz_p1_ci = wilson_rate_ci(zz_p1_count, shots)
            xx_p1_ci = wilson_rate_ci(xx_p1_count, shots)
            
            # Transform CI endpoints: E = 1 - 2p
            zz_ci = (1.0 - 2.0 * zz_p1_ci[1], 1.0 - 2.0 * zz_p1_ci[0])  # Reverse order for E
            xx_ci = (1.0 - 2.0 * xx_p1_ci[1], 1.0 - 2.0 * xx_p1_ci[0])  # Reverse order for E
            
            # Bell fidelity bound
            fidelity_bound = 0.5 * (zz_expectation + xx_expectation)
            
            correlations[pair_key] = {
                "zz_correlator": zz_expectation,
                "xx_correlator": xx_expectation,
                "zz_ci": zz_ci,
                "xx_ci": xx_ci,
                "fidelity_bound": fidelity_bound,
                "zz_operator": zz_data.get("logical_operator"),
                "xx_operator": xx_data.get("logical_operator"),
                "zz_physical": zz_data.get("physical_realization"),
                "xx_physical": xx_data.get("physical_realization"),
            }
    
    return correlations
