
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from qiskit_qec.operators.pauli_list import PauliList

if TYPE_CHECKING:  # pragma: no cover - import guard for type checking only
    from simulation.runner import SimulationResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


@dataclass
class PauliFrameStats:
    """Computed distributions for raw/expected/decoder Pauli frames."""

    expected_frame_bit: int
    decoder_flip_rate: float
    tracked_frame_rate: float
    raw_logicals: np.ndarray
    raw_predictions: np.ndarray
    logical_expected: np.ndarray
    decoded_expected: np.ndarray
    tracked_pauli_frame: np.ndarray
    logical_tracked: np.ndarray
    decoded_tracked: np.ndarray
    logical_probs: Dict[str, Dict[str, float]]
    decoded_probs: Dict[str, Dict[str, float]]
    frame_prob: Dict[str, float]
    logical_means: Dict[str, float]
    decoded_means: Dict[str, float]


def _binary_distribution(mean: float) -> Dict[str, float]:
    return {"|0>": 1.0 - mean, "|1>": mean}


def compute_pauli_frame_stats(
    result: "SimulationResult",
    basis: str,
    expected_flip_total: int,
    column: int = 0,
) -> PauliFrameStats:
    """Return Pauli-frame derived quantities for a given logical basis."""

    raw_logicals = result.logical_observables[:, column]
    raw_predictions = result.predictions[:, column]
    decoder_frame = result.decoder_frame()
    decoder_corrections = decoder_frame.correction_bits(basis, column=column)
    decoder_flip_rate = float(decoder_corrections.mean())

    expected_frame_bit = int(expected_flip_total) & 1
    logical_expected = np.bitwise_xor(raw_logicals, expected_frame_bit)
    decoded_expected = np.bitwise_xor(raw_predictions, expected_frame_bit)

    tracked_pauli_frame = np.bitwise_xor(decoder_corrections, expected_frame_bit)
    tracked_frame_rate = float(tracked_pauli_frame.mean())
    logical_tracked = np.bitwise_xor(raw_logicals, tracked_pauli_frame)
    decoded_tracked = np.bitwise_xor(raw_predictions, tracked_pauli_frame)

    logical_means = {
        "raw": float(raw_logicals.mean()),
        "expected": float(logical_expected.mean()),
        "decoder": float(logical_tracked.mean()),
    }
    decoded_means = {
        "raw": float(raw_predictions.mean()),
        "expected": float(decoded_expected.mean()),
        "decoder": float(decoded_tracked.mean()),
    }

    logical_probs = {
        "raw": _binary_distribution(logical_means["raw"]),
        "expected_frame": _binary_distribution(logical_means["expected"]),
        "decoder_frame": _binary_distribution(logical_means["decoder"]),
    }
    decoded_probs = {
        "raw": _binary_distribution(decoded_means["raw"]),
        "expected_frame": _binary_distribution(decoded_means["expected"]),
        "decoder_frame": _binary_distribution(decoded_means["decoder"]),
    }
    frame_prob = _binary_distribution(tracked_frame_rate)

    return PauliFrameStats(
        expected_frame_bit=expected_frame_bit,
        decoder_flip_rate=decoder_flip_rate,
        tracked_frame_rate=tracked_frame_rate,
        raw_logicals=raw_logicals,
        raw_predictions=raw_predictions,
        logical_expected=logical_expected,
        decoded_expected=decoded_expected,
        tracked_pauli_frame=tracked_pauli_frame,
        logical_tracked=logical_tracked,
        decoded_tracked=decoded_tracked,
        logical_probs=logical_probs,
        decoded_probs=decoded_probs,
        frame_prob=frame_prob,
        logical_means=logical_means,
        decoded_means=decoded_means,
    )


def print_logical_results(
    args,
    gate_seq,
    start_basis,
    init_sign,
    end_basis,
    expected_flip_total,
    stim_rounds,
    result,
    frame_stats: PauliFrameStats,
):
    print("Simulation: logical 1Q sequence on heavy-hex surface code")
    print(f"  benchmark = {args.benchmark}")
    print(f"  sequence  = {' '.join(gate_seq) if gate_seq else '(empty)'}")
    print(f"  init/start basis = {start_basis} (sign {init_sign:+d}), end basis = {end_basis}, expected flip = {expected_flip_total}")
    print(f"  distance = {args.distance}, rounds = {stim_rounds}")
    print(f"  p_x = {args.px}, p_z = {args.pz}")
    print(f"  shots = {result.shots}, detectors = {result.num_detectors}")
    print(f"  logical_error_rate = {result.logical_error_rate:.3e}")
    print(f"  avg_syndrome_weight = {result.avg_syndrome_weight:.3f}")
    print(f"  click_rate = {result.click_rate:.3f}")
    print(f"  decoder_observable parity rate (basis {end_basis}) = {frame_stats.decoder_flip_rate:.3f}")
    print(f"  tracked_pauli_frame parity rate (basis {end_basis}) = {frame_stats.tracked_frame_rate:.3f}")

    """"""

    print("----------------EXPECTED RESULTS WITHOUT PAULI FRAME CORRECTION:----------------")
    print(
        "  logical_raw dist: |0>={:6.2f}% |1>={:6.2f}%".format(
            frame_stats.logical_probs["raw"]["|0>"] * 100.0, frame_stats.logical_probs["raw"]["|1>"] * 100.0
        )
    )
    print(
        "  decoded_raw dist: |0>={:6.2f}% |1>={:6.2f}%".format(
            frame_stats.decoded_probs["raw"]["|0>"] * 100.0,
            frame_stats.decoded_probs["raw"]["|1>"] * 100.0,
        )
    )

    print("----------------EXPECTED FRAME (LOGICAL GATE TRACKING ONLY):----------------")
    print(
        "  logical_expected dist: |0>={:6.2f}% |1>={:6.2f}%".format(
            frame_stats.logical_probs["expected_frame"]["|0>"] * 100.0,
            frame_stats.logical_probs["expected_frame"]["|1>"] * 100.0,
        )
    )
    print(
        "  decoded_expected dist: |0>={:6.2f}% |1>={:6.2f}%".format(
            frame_stats.decoded_probs["expected_frame"]["|0>"] * 100.0,
            frame_stats.decoded_probs["expected_frame"]["|1>"] * 100.0,
        )
    )

    print("----------------TRACKED PAULI FRAME (GATES + DECODER):----------------")

    print(
        "  logical_post_correction dist: |0>={:6.2f}% |1>={:6.2f}%".format(
            frame_stats.logical_probs["decoder_frame"]["|0>"] * 100.0,
            frame_stats.logical_probs["decoder_frame"]["|1>"] * 100.0,
        )
    )
   
