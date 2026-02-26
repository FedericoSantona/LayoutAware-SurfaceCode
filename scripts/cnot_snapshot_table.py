"""Run CNOT snapshot sweeps from an IBM fake-backend calibration.

This script:
1. Builds a calibration snapshot from a selected fake backend.
2. Uses case-specific qubit mappings for each (layout, distance) run.
3. Runs CNOT logical-error simulations with mapped device-aware noise.
4. Prints a results table and saves it to CSV/Markdown.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from math import gcd
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List

# Ensure src/ is importable when executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from simulation import run_cnot_logical_error_rate
from surface_code import (
    CouplerNoiseParams,
    DeviceCalibration,
    QubitNoiseParams,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fake-backend",
        type=str,
        default="FakeMarrakesh",
        help="Class name from qiskit_ibm_runtime.fake_provider (e.g. FakeMarrakesh).",
    )
    parser.add_argument(
        "--distances",
        nargs="*",
        type=int,
        default=[3, 5, 7, 9],
        help="Code distances to run (default: 3 5 7 9).",
    )
    parser.add_argument(
        "--layouts",
        nargs="*",
        choices=["heavy_hex", "standard"],
        default=["heavy_hex", "standard"],
        help="Layouts to run (default: heavy_hex standard).",
    )
    parser.add_argument("--shots", type=int, default=1000000, help="Shots per case.")
    parser.add_argument("--seed", type=int, default=46, help="Random seed.")
    parser.add_argument(
        "--round-duration",
        type=float,
        default=1.0,
        help="Round duration in microseconds for p_x/p_z mapping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "cnot_snapshot",
        help="Directory where outputs are written.",
    )
    return parser.parse_args()


def _instantiate_fake_backend(fake_backend_name: str) -> Any:
    from qiskit_ibm_runtime import fake_provider

    backend_cls = getattr(fake_provider, fake_backend_name, None)
    if backend_cls is None:
        available = sorted(n for n in dir(fake_provider) if n.startswith("Fake"))
        preview = ", ".join(available[:20])
        raise ValueError(
            f"Unknown fake backend '{fake_backend_name}'. Example available names: {preview}"
        )
    return backend_cls()


def _format_table(rows: Iterable[Dict[str, Any]], columns: List[str]) -> str:
    formatted: List[List[str]] = []
    for row in rows:
        formatted.append([str(row.get(col, "")) for col in columns])

    widths = [len(col) for col in columns]
    for values in formatted:
        for i, val in enumerate(values):
            widths[i] = max(widths[i], len(val))

    def fmt_line(values: List[str]) -> str:
        parts = [values[i].ljust(widths[i]) for i in range(len(values))]
        return " | ".join(parts)

    header = fmt_line(columns)
    sep = "-+-".join("-" * w for w in widths)
    body = "\n".join(fmt_line(v) for v in formatted)
    return f"{header}\n{sep}\n{body}"


def _to_markdown(rows: Iterable[Dict[str, Any]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        vals = [str(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _calibration_ranges(calibration: DeviceCalibration) -> Dict[str, float]:
    t1_values = [p.t1 for p in calibration.qubit_params.values()]
    t2_values = [p.t2 for p in calibration.qubit_params.values()]
    readout_values = [p.readout_error for p in calibration.qubit_params.values()]
    gate_1q_values = [p.single_qubit_gate_error for p in calibration.qubit_params.values()]
    gate_2q_values = [p.two_qubit_gate_error for p in calibration.coupler_params.values()]

    out = {
        "t1_min_us": min(t1_values),
        "t1_max_us": max(t1_values),
        "t2_min_us": min(t2_values),
        "t2_max_us": max(t2_values),
        "readout_min": min(readout_values),
        "readout_max": max(readout_values),
        "gate1q_min": min(gate_1q_values),
        "gate1q_max": max(gate_1q_values),
    }
    if gate_2q_values:
        out["gate2q_min"] = min(gate_2q_values)
        out["gate2q_max"] = max(gate_2q_values)
    else:
        out["gate2q_min"] = float("nan")
        out["gate2q_max"] = float("nan")
    return out


def _required_simulation_qubits(distance: int, layout: str) -> int:
    """Total qubits for the 3-patch CNOT layout at (distance, layout)."""
    if layout == "heavy_hex":
        # In this repository's heavy-hex implementation, one patch uses d^2 data qubits.
        n_single = distance * distance
    elif layout == "standard":
        n_single = 2 * distance * distance - 2 * distance + 1
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    return 3 * n_single + 2 * distance


def _find_coprime_stride(total: int, preferred: int) -> int:
    stride = max(1, preferred)
    while gcd(stride, total) != 1:
        stride += 1
    return stride


def _case_seed(layout: str, distance: int) -> int:
    return distance * 101 + sum(ord(ch) for ch in layout)


def _select_connected_subset(
    calibration: DeviceCalibration,
    ordered_qubits: List[int],
    required: int,
    anchor: int,
) -> List[int] | None:
    """Try to choose a connected qubit subset when coupler data is available."""
    if not calibration.coupler_params:
        return None

    adjacency: Dict[int, set[int]] = {q: set() for q in ordered_qubits}
    for (q1, q2) in calibration.coupler_params:
        if q1 in adjacency and q2 in adjacency:
            adjacency[q1].add(q2)
            adjacency[q2].add(q1)

    if anchor not in adjacency:
        return None

    queue: deque[int] = deque([anchor])
    visited: set[int] = {anchor}
    bfs_order: List[int] = []

    while queue and len(bfs_order) < required:
        q = queue.popleft()
        bfs_order.append(q)
        for nb in sorted(adjacency[q]):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    if len(bfs_order) < required:
        for q in ordered_qubits:
            if q not in visited:
                bfs_order.append(q)
                visited.add(q)
                if len(bfs_order) >= required:
                    break

    if len(bfs_order) < required:
        return None
    return bfs_order[:required]


def _select_backend_qubits_for_case(
    calibration: DeviceCalibration,
    layout: str,
    distance: int,
    required: int,
) -> tuple[List[int], str]:
    """Select backend qubits for one case.

    Returns:
        (selected_backend_qubits, mapping_mode)
    """
    ordered = sorted(calibration.qubit_indices)
    total = len(ordered)
    if total == 0:
        raise ValueError("Calibration contains no qubits.")

    seed = _case_seed(layout, distance)
    start = seed % total
    preferred_stride = 1 if layout == "heavy_hex" else 5
    stride = _find_coprime_stride(total, preferred_stride)

    if required <= total:
        anchor = ordered[start]
        connected = _select_connected_subset(calibration, ordered, required, anchor)
        if connected is not None:
            return connected, "unique_connected"

        selected: List[int] = []
        seen: set[int] = set()
        idx = start
        while len(selected) < required:
            q = ordered[idx]
            if q not in seen:
                selected.append(q)
                seen.add(q)
            idx = (idx + stride) % total
        return selected, "unique_strided"

    # For large code distances where the simulated CNOT layout needs more
    # qubits than the calibration snapshot provides, reuse backend qubits
    # deterministically to preserve heterogeneity statistics.
    selected = [ordered[(start + i * stride) % total] for i in range(required)]
    return selected, "reused_strided"


def _copy_qubit_params(params: QubitNoiseParams) -> QubitNoiseParams:
    return QubitNoiseParams(
        t1=params.t1,
        t2=params.t2,
        readout_error_0to1=params.readout_error_0to1,
        readout_error_1to0=params.readout_error_1to0,
        single_qubit_gate_error=params.single_qubit_gate_error,
        frequency=params.frequency,
    )


def _copy_coupler_params(params: CouplerNoiseParams) -> CouplerNoiseParams:
    return CouplerNoiseParams(
        two_qubit_gate_error=params.two_qubit_gate_error,
        crosstalk_strength=params.crosstalk_strength,
    )


def _build_case_calibration(
    base: DeviceCalibration,
    selected_backend_qubits: List[int],
    case_name: str,
    mapping_mode: str,
) -> DeviceCalibration:
    """Remap backend calibration onto local simulation qubit indices 0..n-1."""
    qubit_params: Dict[int, QubitNoiseParams] = {}
    for local_q, backend_q in enumerate(selected_backend_qubits):
        qubit_params[local_q] = _copy_qubit_params(base.qubit_params[backend_q])

    coupler_params: Dict[tuple[int, int], CouplerNoiseParams] = {}
    unique_backend = len(set(selected_backend_qubits)) == len(selected_backend_qubits)
    if unique_backend:
        inv = {backend_q: local_q for local_q, backend_q in enumerate(selected_backend_qubits)}
        for (q1, q2), edge_params in base.coupler_params.items():
            if q1 in inv and q2 in inv:
                local_edge = (min(inv[q1], inv[q2]), max(inv[q1], inv[q2]))
                coupler_params[local_edge] = _copy_coupler_params(edge_params)

    metadata = dict(base.metadata)
    metadata.update(
        {
            "source_calibration_backend": base.backend_name,
            "mapping_case": case_name,
            "mapping_mode": mapping_mode,
            "mapping_selected_backend_qubits": selected_backend_qubits,
            "mapping_unique_backend_qubits": sorted(set(selected_backend_qubits)),
        }
    )

    return DeviceCalibration(
        backend_name=f"{base.backend_name}:{case_name}",
        timestamp=base.timestamp,
        qubit_params=qubit_params,
        coupler_params=coupler_params,
        gate_times=dict(base.gate_times),
        metadata=metadata,
    )


def _mapped_rates(
    calibration: DeviceCalibration,
    round_duration: float,
) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    noise_model = calibration.to_noise_model(default_round_duration=round_duration)
    qubits = calibration.qubit_indices
    px_values = [noise_model.get_effective_error_rate(q, "x") for q in qubits]
    pz_values = [noise_model.get_effective_error_rate(q, "z") for q in qubits]
    return (
        fmean(px_values),
        fmean(pz_values),
        (min(px_values), max(px_values)),
        (min(pz_values), max(pz_values)),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    backend = _instantiate_fake_backend(args.fake_backend)
    base_calibration = DeviceCalibration.from_ibm_backend(backend)
    machine_name = base_calibration.backend_name
    initial_ranges = _calibration_ranges(base_calibration)

    calibration_path = args.output_dir / f"{machine_name}_calibration_snapshot.json"
    base_calibration.to_json(calibration_path)

    print(f"Fake machine: {machine_name}")
    print(f"Calibration snapshot saved to: {calibration_path}")

    rows: List[Dict[str, Any]] = []
    total = len(args.layouts) * len(args.distances)
    done = 0

    for layout in args.layouts:
        for distance in args.distances:
            done += 1
            required_qubits = _required_simulation_qubits(distance, layout)
            selected_backend_qubits, mapping_mode = _select_backend_qubits_for_case(
                base_calibration,
                layout,
                distance,
                required_qubits,
            )
            case_name = f"{layout}_d{distance}"
            case_calibration = _build_case_calibration(
                base=base_calibration,
                selected_backend_qubits=selected_backend_qubits,
                case_name=case_name,
                mapping_mode=mapping_mode,
            )

            mapped_px, mapped_pz, mapped_px_range, mapped_pz_range = _mapped_rates(
                case_calibration,
                args.round_duration,
            )
            case_noise_model = case_calibration.to_noise_model(
                default_round_duration=args.round_duration,
            )

            print(
                f"[{done}/{total}] layout={layout}, d={distance}, shots={args.shots}, "
                f"map={mapping_mode}, sim_qubits={required_qubits}, "
                f"backend_qubits_used={len(set(selected_backend_qubits))}/{len(selected_backend_qubits)}"
            )

            result = run_cnot_logical_error_rate(
                distance=distance,
                code_type=layout,
                p_x=0.0,
                p_z=0.0,
                shots=args.shots,
                seed=args.seed,
                rounds_pre=distance,
                rounds_merge=distance,
                rounds_post=distance,
                noise_model=case_noise_model,
                verbose=False,
            )

            control_ler = (
                result.logical_error_rates[0]
                if len(result.logical_error_rates) > 0
                else result.logical_error_rate
            )
            target_ler = (
                result.logical_error_rates[1]
                if len(result.logical_error_rates) > 1
                else result.logical_error_rate
            )
            avg_ler = 0.5 * (control_ler + target_ler)

            rows.append(
                {
                    "fake_machine": machine_name,
                    "layout": layout,
                    "distance": distance,
                    "shots": args.shots,
                    "mapping_mode": mapping_mode,
                    "sim_qubits_required": required_qubits,
                    "backend_qubits_unique_used": len(set(selected_backend_qubits)),
                    "initial_t1_range_us": f"{_fmt(initial_ranges['t1_min_us'])}-{_fmt(initial_ranges['t1_max_us'])}",
                    "initial_t2_range_us": f"{_fmt(initial_ranges['t2_min_us'])}-{_fmt(initial_ranges['t2_max_us'])}",
                    "initial_readout_range": f"{_fmt(initial_ranges['readout_min'])}-{_fmt(initial_ranges['readout_max'])}",
                    "initial_gate1q_range": f"{_fmt(initial_ranges['gate1q_min'])}-{_fmt(initial_ranges['gate1q_max'])}",
                    "initial_gate2q_range": f"{_fmt(initial_ranges['gate2q_min'])}-{_fmt(initial_ranges['gate2q_max'])}",
                    "mapped_px": f"{mapped_px:.6e}",
                    "mapped_pz": f"{mapped_pz:.6e}",
                    "mapped_px_range": f"{mapped_px_range[0]:.6e}-{mapped_px_range[1]:.6e}",
                    "mapped_pz_range": f"{mapped_pz_range[0]:.6e}-{mapped_pz_range[1]:.6e}",
                    "logical_error_rate_control": f"{control_ler:.6e}",
                    "logical_error_rate_target": f"{target_ler:.6e}",
                    "logical_error_rate_avg": f"{avg_ler:.6e}",
                }
            )

    columns = [
        "fake_machine",
        "layout",
        "distance",
        "shots",
        "mapping_mode",
        "sim_qubits_required",
        "backend_qubits_unique_used",
        "initial_t1_range_us",
        "initial_t2_range_us",
        "initial_readout_range",
        "initial_gate1q_range",
        "initial_gate2q_range",
        "mapped_px",
        "mapped_pz",
        "mapped_px_range",
        "mapped_pz_range",
        "logical_error_rate_control",
        "logical_error_rate_target",
        "logical_error_rate_avg",
    ]

    print("\nResults table:\n")
    print(_format_table(rows, columns))

    csv_path = args.output_dir / f"cnot_snapshot_table_{machine_name}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.output_dir / f"cnot_snapshot_table_{machine_name}.md"
    md_path.write_text(_to_markdown(rows, columns), encoding="utf-8")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
