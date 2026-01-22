"""Run threshold sweeps and generate plots for the heavy-hex surface code."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Ensure src/ is importable when executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_PATH))

# Use a local Matplotlib cache to avoid permission issues
mpl_cache = PROJECT_ROOT / ".mplconfig"
mpl_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

try:  # Prefer a progress bar when available, but don't fail without it
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None  # type: ignore[assignment]

from simulation.code_threshold import (
    EXPERIMENT_TYPE_CNOT,
    EXPERIMENT_TYPE_MEMORY,
    ThresholdScenario,
    ThresholdScenarioResult,
    ThresholdStudyConfig,
    create_standard_scenarios,
    estimate_crossings,
    estimate_threshold,
    run_scenario,
    export_csv,
    plot_scenario,
)
from surface_code import (
    DeviceCalibration,
    NoiseModel,
    PhenomenologicalNoiseModel,
)


def _logical_metadata(track: str | None) -> tuple[str | None, str | None]:
    """Return (logical_measured, physical_sensitivity) for a scenario track."""
    logical = track if track in {"X", "Z"} else None
    if logical == "Z":
        return logical, "X"
    if logical == "X":
        return logical, "Z"
    return logical, None


def parse_distances(values: Sequence[str]) -> list[int]:
    if not values:
        return [3, 5, 7, 9]
    return [int(v) for v in values]


def make_physical_grid(p_min: float, p_max: float, num: int) -> np.ndarray:
    return np.logspace(np.log10(p_min), np.log10(p_max), num=num)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=10**4, help="Monte Carlo shots per data point")
    parser.add_argument("--seed", type=int, default=46, help="Random seed for Stim samplers")
    parser.add_argument(
        "--distances",
        nargs="*",
        default=[3, 5, 7],
        help="Code distances to include (default: 3 5 7 9)",
    )
    parser.add_argument("--p-min", type=float, default=5e-4, help="Minimum physical error rate")
    parser.add_argument("--p-max", type=float, default=0.1, help="Maximum physical error rate")
    parser.add_argument("--num-points", type=int, default=20, help="Number of physical error samples")
    parser.add_argument(
        "--layout",
        type=str,
        choices=["heavy_hex", "standard"],
        default="heavy_hex",
        help="Surface code layout type: 'heavy_hex' or 'standard' (default: heavy_hex)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory to store generated plots (default: plots/threshold/{layout}/{experiment_type})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory to store CSV/JSON results (default: output/threshold/{layout}/{experiment_type})",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        choices=[EXPERIMENT_TYPE_MEMORY, EXPERIMENT_TYPE_CNOT],
        default=EXPERIMENT_TYPE_CNOT,
        help=f"Experiment type: '{EXPERIMENT_TYPE_MEMORY}' for memory threshold or "
             f"'{EXPERIMENT_TYPE_CNOT}' for CNOT lattice-surgery threshold (default: {EXPERIMENT_TYPE_MEMORY})",
    )
    # Noise model options
    parser.add_argument(
        "--noise-model",
        type=str,
        choices=["phenomenological", "device-aware"],
        default="phenomenological",
        help="Noise model type: 'phenomenological' (uniform p_x/p_z) or "
             "'device-aware' (per-qubit T1/T2 from calibration)",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=None,
        help="Path to device calibration JSON file (required for device-aware noise model)",
    )
    parser.add_argument(
        "--round-duration",
        type=float,
        default=1.0,
        help="Measurement round duration in microseconds (for device-aware noise, default: 1.0)",
    )
    return parser.parse_args()


def load_noise_model(args: argparse.Namespace) -> Optional[NoiseModel]:
    """Load and configure the noise model based on CLI arguments.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        NoiseModel instance, or None for legacy phenomenological mode.
    """
    if args.noise_model == "phenomenological":
        # Return None to use the legacy p_x_error/p_z_error approach
        # (noise rates are set per data point in the sweep)
        return None
    
    elif args.noise_model == "device-aware":
        if args.calibration_file is None:
            print("Error: --calibration-file is required for device-aware noise model")
            print("  Use --calibration-file <path> to specify device calibration JSON")
            sys.exit(1)
        
        if not args.calibration_file.exists():
            print(f"Error: Calibration file not found: {args.calibration_file}")
            sys.exit(1)
        
        # Load calibration and create noise model
        print(f"Loading device calibration from: {args.calibration_file}")
        calibration = DeviceCalibration.from_json(args.calibration_file)
        print(calibration.summary())
        
        noise_model = calibration.to_noise_model(
            default_round_duration=args.round_duration,
        )
        return noise_model
    
    else:
        raise ValueError(f"Unknown noise model type: {args.noise_model}")


def scenario_to_dict(result: ThresholdScenarioResult) -> dict:
    logical, physical = _logical_metadata(result.track)
    data = {
        "name": result.name,
        "init_label": result.init_label,
        "track": result.track,
        "logical": logical,
        "sensitive_to": physical,
        "distances": result.distances,
        "sweeps": [],
    }
    for sweep in result.sweeps:
        sweep_data = {
            "distance": sweep.distance,
            "points": [
                {
                    "p_x": point.p_x,
                    "p_z": point.p_z,
                    "logical_error_rate": point.logical_error_rate,
                    "avg_syndrome_weight": point.avg_syndrome_weight,
                    "click_rate": point.click_rate,
                }
                for point in sweep.points
            ],
        }
        data["sweeps"].append(sweep_data)
    return data


def main() -> None:
    args = parse_args()
    distances = parse_distances(args.distances or [])
    physical_grid = make_physical_grid(args.p_min, args.p_max, args.num_points)

    # Set default directories based on layout type and experiment type if not provided
    plot_dir: Path = args.plot_dir or (PROJECT_ROOT / "plots" / "threshold" / args.layout / args.experiment_type)
    data_dir: Path = args.data_dir or (PROJECT_ROOT / "output" / "threshold" / args.layout / args.experiment_type)
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load noise model if device-aware mode is selected
    noise_model = load_noise_model(args)
    if noise_model is not None:
        print(f"Using device-aware noise model with round duration {args.round_duration} µs")
    else:
        print("Using phenomenological noise model (uniform error rates)")

    scenarios = create_standard_scenarios(distances, physical_grid)
    study_cfg = ThresholdStudyConfig(shots=args.shots, seed=args.seed)
    # Ignore sign flips that occur while logical error rates are below the MC noise floor.
    noise_floor = 5 / study_cfg.shots if study_cfg.shots else None

    summary = {}
    use_tqdm = tqdm is not None and sys.stdout.isatty()

    for scenario in scenarios:
        total_runs = len(scenario.distances) * len(scenario.physical_error_grid)
        progress_bar = None
        if use_tqdm and total_runs > 0:
            progress_bar = tqdm(
                total=total_runs,
                desc=f"{scenario.name} sweeps",
                unit="run",
                dynamic_ncols=True,
            )

        def log(message: str) -> None:
            if progress_bar is not None:
                progress_bar.write(message)
            else:
                print(message)

        def update_progress(scenario: ThresholdScenario, distance: int, p_x: float, p_z: float) -> None:
            if progress_bar is None:
                return
            progress_bar.set_postfix_str(
                f"d={distance} p_x={p_x:.1e} p_z={p_z:.1e}",
                refresh=False,
            )
            progress_bar.update()

        log(f"Running scenario {scenario.name} (init {scenario.init_label}) [{args.experiment_type}]")
        logical, physical = _logical_metadata(scenario.track)
        if logical:
            log(f"  measuring logical {logical}, primarily sensitive to physical {physical} errors")
        result = run_scenario(
            scenario,
            study_cfg,
            progress=update_progress if progress_bar is not None else None,
            code_type=args.layout,
            experiment_type=args.experiment_type,
            noise_model=noise_model,
        )
        csv_paths = export_csv(result, data_dir)
        plot_path = plot_scenario(result, plot_dir ,1 / args.shots)
        crossings = estimate_crossings(result, min_logical_error_rate=noise_floor)
        threshold_est = estimate_threshold(crossings)
        logical, physical = _logical_metadata(result.track)
        
        # Build threshold estimate dict for JSON output
        threshold_dict = {
            "best_estimate": threshold_est.best_estimate,
            "weighted_average": threshold_est.weighted_average,
            "simple_average": threshold_est.simple_average,
            "num_crossings": threshold_est.num_crossings,
            "highest_distance_pair": (
                f"{threshold_est.highest_distance_pair[0]}-{threshold_est.highest_distance_pair[1]}"
                if threshold_est.highest_distance_pair else None
            ),
        }
        
        summary[scenario.name] = {
            "experiment_type": args.experiment_type,
            "csv": {str(distance): str(path.relative_to(PROJECT_ROOT)) for distance, path in csv_paths.items()},
            "plot": str(plot_path.relative_to(PROJECT_ROOT)),
            "logical": logical,
            "sensitive_to": physical,
            "crossings": {f"{d1}-{d2}": crossing for (d1, d2), crossing in crossings.items()},
            "threshold_estimate": threshold_dict,
        }
        
        # Include threshold estimate and experiment type in per-scenario JSON
        scenario_data = scenario_to_dict(result)
        scenario_data["experiment_type"] = args.experiment_type
        scenario_data["threshold_estimate"] = threshold_dict
        
        json_path = data_dir / f"{scenario.name}.json"
        with json_path.open("w") as fh:
            json.dump(scenario_data, fh, indent=2)
        summary[scenario.name]["json"] = str(json_path.relative_to(PROJECT_ROOT))
        
        for pair, crossing in crossings.items():
            d1, d2 = pair
            if crossing is None:
                log(f"  d={d1} vs d={d2}: no crossing within sampled range")
            else:
                log(f"  d={d1} vs d={d2}: estimated crossing at p ~ {crossing:.3e}")
        
        # Log the threshold estimate
        if threshold_est.best_estimate is not None:
            log(f"  THRESHOLD ESTIMATE: p ~ {threshold_est.best_estimate:.4e} "
                f"(from d={threshold_est.highest_distance_pair[0]} vs d={threshold_est.highest_distance_pair[1]})")
        else:
            log(f"  THRESHOLD ESTIMATE: could not be determined (no valid crossings)")
        
        log(f"  plot saved to {plot_path}")

        if progress_bar is not None:
            progress_bar.close()

    # Add top-level metadata to the summary
    noise_metadata = {
        "noise_model": args.noise_model,
    }
    if args.noise_model == "device-aware" and args.calibration_file:
        noise_metadata["calibration_file"] = str(args.calibration_file)
        noise_metadata["round_duration_us"] = args.round_duration
    
    full_summary = {
        "_metadata": {
            "experiment_type": args.experiment_type,
            "layout": args.layout,
            "distances": distances,
            "shots": args.shots,
            "p_min": args.p_min,
            "p_max": args.p_max,
            "num_points": args.num_points,
            **noise_metadata,
        },
        "scenarios": summary,
    }
    
    summary_path = data_dir / "threshold_summary.json"
    with summary_path.open("w") as fh:
        json.dump(full_summary, fh, indent=2)

    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
