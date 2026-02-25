"""Run CNOT snapshot sweeps from an IBM fake-backend calibration.

This script:
1. Builds a calibration snapshot from a selected fake backend.
2. Derives effective uniform p_x/p_z from the device-aware noise model.
3. Runs CNOT logical-error simulations for (distance, layout) combinations.
4. Prints a results table and saves it to CSV/Markdown.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List

# Ensure src/ is importable when executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from simulation import run_cnot_logical_error_rate
from surface_code import DeviceCalibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fake-backend",
        type=str,
        default="FakeSherbrooke",
        help="Class name from qiskit_ibm_runtime.fake_provider (e.g. FakeSherbrooke).",
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
    parser.add_argument("--shots", type=int, default=10000, help="Shots per case.")
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


def _derive_px_pz(
    calibration: DeviceCalibration,
    round_duration: float,
) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    noise_model = calibration.to_noise_model(default_round_duration=round_duration)
    qubits = calibration.qubit_indices
    if not qubits:
        raise ValueError("Calibration contains no qubits.")

    px_values = [noise_model.get_effective_error_rate(q, "x") for q in qubits]
    pz_values = [noise_model.get_effective_error_rate(q, "z") for q in qubits]
    return (
        fmean(px_values),
        fmean(pz_values),
        (min(px_values), max(px_values)),
        (min(pz_values), max(pz_values)),
    )


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    backend = _instantiate_fake_backend(args.fake_backend)
    calibration = DeviceCalibration.from_ibm_backend(backend)
    machine_name = calibration.backend_name

    calibration_path = args.output_dir / f"{machine_name}_calibration_snapshot.json"
    calibration.to_json(calibration_path)

    px, pz, px_range, pz_range = _derive_px_pz(calibration, args.round_duration)
    ranges = _calibration_ranges(calibration)

    print(f"Fake machine: {machine_name}")
    print(f"Calibration snapshot saved to: {calibration_path}")
    print(f"Mapped p_x={px:.6e} (min={px_range[0]:.6e}, max={px_range[1]:.6e})")
    print(f"Mapped p_z={pz:.6e} (min={pz_range[0]:.6e}, max={pz_range[1]:.6e})")

    rows: List[Dict[str, Any]] = []
    total = len(args.layouts) * len(args.distances)
    done = 0
    for layout in args.layouts:
        for distance in args.distances:
            done += 1
            print(f"[{done}/{total}] layout={layout}, d={distance}, shots={args.shots}")
            result = run_cnot_logical_error_rate(
                distance=distance,
                code_type=layout,
                p_x=px,
                p_z=pz,
                shots=args.shots,
                seed=args.seed,
                rounds_pre=distance,
                rounds_merge=distance,
                rounds_post=distance,
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

            row = {
                "fake_machine": machine_name,
                "layout": layout,
                "distance": distance,
                "shots": args.shots,
                "num_qubits": calibration.num_qubits,
                "num_couplers": len(calibration.coupler_params),
                "t1_range_us": f"{_fmt(ranges['t1_min_us'])}-{_fmt(ranges['t1_max_us'])}",
                "t2_range_us": f"{_fmt(ranges['t2_min_us'])}-{_fmt(ranges['t2_max_us'])}",
                "readout_range": f"{_fmt(ranges['readout_min'])}-{_fmt(ranges['readout_max'])}",
                "gate1q_range": f"{_fmt(ranges['gate1q_min'])}-{_fmt(ranges['gate1q_max'])}",
                "gate2q_range": f"{_fmt(ranges['gate2q_min'])}-{_fmt(ranges['gate2q_max'])}",
                "mapped_px": f"{px:.6e}",
                "mapped_pz": f"{pz:.6e}",
                "logical_error_rate_control": f"{control_ler:.6e}",
                "logical_error_rate_target": f"{target_ler:.6e}",
                "logical_error_rate_avg": f"{avg_ler:.6e}",
                "avg_syndrome_weight": f"{result.avg_syndrome_weight:.6f}",
                "click_rate": f"{result.click_rate:.6f}",
            }
            rows.append(row)

    columns = [
        "fake_machine",
        "layout",
        "distance",
        "shots",
        "num_qubits",
        "num_couplers",
        "t1_range_us",
        "t2_range_us",
        "readout_range",
        "gate1q_range",
        "gate2q_range",
        "mapped_px",
        "mapped_pz",
        "logical_error_rate_control",
        "logical_error_rate_target",
        "logical_error_rate_avg",
    ]

    print("\nResults table:\n")
    print(_format_table(rows, columns))

    csv_path = args.output_dir / f"cnot_snapshot_table_{machine_name}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns + ["avg_syndrome_weight", "click_rate"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.output_dir / f"cnot_snapshot_table_{machine_name}.md"
    md_path.write_text(_to_markdown(rows, columns), encoding="utf-8")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
