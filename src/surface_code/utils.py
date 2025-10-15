
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
    override_raw: np.ndarray | None = None,
    apply_decoder: bool = True,
) -> PauliFrameStats:
    """Return Pauli-frame derived quantities for a given logical basis."""

    raw_logicals = result.logical_observables[:, column] if override_raw is None else np.asarray(override_raw, dtype=np.uint8)
    raw_predictions = result.predictions[:, column]
    decoder_frame = result.decoder_frame()
    if apply_decoder:
        decoder_corrections = decoder_frame.correction_bits(basis, column=column)
    else:
        decoder_corrections = np.zeros_like(raw_logicals, dtype=np.uint8)
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


def print_multi_qubit_results(
    args,
    basis_labels: tuple[str, ...],
    stim_rounds: int,
    result: "SimulationResult",
    expected_flips: "np.ndarray | list[int] | tuple[int, ...] | None" = None,
    gate_seq: "list[str] | None" = None,
):
    """Print per-qubit logical distributions in the style of print_logical_results.

    For each observable column, prints the raw/expected/decoder distributions.
    Expected-frame bit defaults to 0 in this multi-qubit summary unless the
    caller pre-adjusts the SimulationResult or passes override bits elsewhere.

    """
    print("Simulation: logical sequence on heavy-hex surface code")
    print(f"  benchmark = {args.benchmark}")
    if gate_seq is not None:
        print(f"  sequence  = {' '.join(gate_seq) if gate_seq else '(empty)'}")
    num_cols = len(basis_labels)
    print(f"  qubits = {num_cols}, distance = {args.distance}, rounds = {stim_rounds}")
    print(f"  p_x = {args.px}, p_z = {args.pz}")
    print(f"  shots = {result.shots}, detectors = {result.num_detectors}")
    # Decoder logical error rate: mean over columns of XOR(predictions, observables)
    try:
        preds = np.asarray(result.predictions, dtype=np.uint8)
        obs = np.asarray(result.logical_observables, dtype=np.uint8)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        min_cols = min(preds.shape[1], obs.shape[1])
        if min_cols > 0:
            per_qubit_ler = (np.bitwise_xor(preds[:, :min_cols], obs[:, :min_cols]).mean(axis=0)).astype(float)
            avg_ler = float(per_qubit_ler.mean())
            print(f"  decoder logical_error_rate (avg over qubits) = {avg_ler:.3e}")
        else:
            per_qubit_ler = None
    except Exception:
        per_qubit_ler = None
    print("----------------PER-QUBIT LOGICAL RESULTS----------------")
    for idx, basis in enumerate(basis_labels):
        ef = 0
        if expected_flips is not None:
            try:
                ef = int(expected_flips[idx]) & 1
            except Exception:
                ef = 0
        # Ensure column index is within bounds
        column_idx = min(idx, result.logical_observables.shape[1] - 1) if result.logical_observables.shape[1] > 0 else 0
        stats = compute_pauli_frame_stats(
            result,
            basis=basis,
            expected_flip_total=ef,
            column=column_idx,
            override_raw=None,
            apply_decoder=True,
        )
        qlabel = f"Q{idx+1} (basis {basis})"
        print(f"{qlabel} -- raw")
        print(
            "  logical_raw dist: |0>={:6.2f}% |1>={:6.2f}%".format(
                stats.logical_probs["raw"]["|0>"] * 100.0,
                stats.logical_probs["raw"]["|1>"] * 100.0,
            )
        )
        print(
            "  decoded_raw dist: |0>={:6.2f}% |1>={:6.2f}%".format(
                stats.decoded_probs["raw"]["|0>"] * 100.0,
                stats.decoded_probs["raw"]["|1>"] * 100.0,
            )
        )
        print(f"{qlabel} -- expected frame")
        print(
            "  logical_expected dist: |0>={:6.2f}% |1>={:6.2f}%".format(
                stats.logical_probs["expected_frame"]["|0>"] * 100.0,
                stats.logical_probs["expected_frame"]["|1>"] * 100.0,
            )
        )
        print(
            "  decoded_expected dist: |0>={:6.2f}% |1>={:6.2f}%".format(
                stats.decoded_probs["expected_frame"]["|0>"] * 100.0,
                stats.decoded_probs["expected_frame"]["|1>"] * 100.0,
            )
        )
        print(f"{qlabel} -- tracked Pauli frame (decoder)")
        print(
            "  logical_post_correction dist: |0>={:6.2f}% |1>={:6.2f}%".format(
                stats.logical_probs["decoder_frame"]["|0>"] * 100.0,
                stats.logical_probs["decoder_frame"]["|1>"] * 100.0,
            )
        )
        if per_qubit_ler is not None and idx < len(per_qubit_ler):
            print(f"  decoder logical_error_rate = {per_qubit_ler[idx]:.3e}")
