"""Bootstrap uncertainty estimates for threshold crossings from saved sweep data.

This script post-processes saved `output/threshold/**/threshold_summary.json` and
per-scenario JSON files produced by `scripts/threshold_experiment.py`.

It does NOT rerun Stim simulations. Instead it performs a parametric bootstrap:
for each (distance, p) point with logical error-rate estimate p_hat = k/N, it
resamples k' ~ Binomial(N, p_hat) and recomputes the threshold estimator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np

# Ensure src/ is importable when executed directly (match threshold_experiment.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_PATH))

from simulation.code_threshold.threshold import (  # noqa: E402
    DistanceSweepResult,
    ThresholdEstimate,
    ThresholdPoint,
    ThresholdScenarioResult,
    estimate_threshold,
)

EstimatorName = Literal["best", "weighted", "simple"]
CrossingMode = Literal["neighbors", "all_pairs"]

# Named presets for common threshold summary files
PRESET_PATHS = {
    "heavy_hex_memory": PROJECT_ROOT / "output" / "threshold" / "heavy_hex" / "memory" / "threshold_summary.json",
    "heavy_hex_cnot": PROJECT_ROOT / "output" / "threshold" / "heavy_hex" / "cnot" / "threshold_summary.json",
    "standard_memory": PROJECT_ROOT / "output" / "threshold" / "standard" / "memory" / "threshold_summary.json",
    "standard_cnot": PROJECT_ROOT / "output" / "threshold" / "standard" / "cnot" / "threshold_summary.json",
}
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preset",
        choices=list(PRESET_PATHS.keys()),
        default="heavy_hex_memory",
        help=f"Named preset to use (choices: {', '.join(PRESET_PATHS.keys())}). ",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario name to process (e.g. x_only). Default: all scenarios in the summary.",
    )
    p.add_argument("--B", type=int, default=2000, help="Bootstrap replicates (default: 2000).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for bootstrap (default: 0).")
    p.add_argument(
        "--estimator",
        choices=["best", "weighted", "simple"],
        default="simple",
        help="Which threshold estimator to bootstrap (default: best).",
    )
    p.add_argument(
        "--crossings",
        choices=["neighbors", "all_pairs"],
        default="all_pairs",
        help="Crossing pairs to use (default: neighbors).",
    )
    p.add_argument(
        "--min-logical-error-rate",
        type=float,
        default=None,
        help="Noise-floor filter for crossings. Default: 5/shots from summary metadata.",
    )
    p.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level for interval (default: 0.95 -> 2.5/97.5 percentiles).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a JSON report. If omitted, prints a table.",
    )
    return p.parse_args()


def _default_summary_path() -> Optional[Path]:
    """Return first existing preset path, or None."""
    for path in PRESET_PATHS.values():
        if path.exists():
            return path
    return None


def _load_json(path: Path) -> Any:
    with path.open("r") as fh:
        return json.load(fh)


def _scenario_result_from_json(data: dict) -> ThresholdScenarioResult:
    sweeps: list[DistanceSweepResult] = []
    for sweep in data["sweeps"]:
        points: list[ThresholdPoint] = []
        for pt in sweep["points"]:
            points.append(
                ThresholdPoint(
                    p_x=float(pt["p_x"]),
                    p_z=float(pt["p_z"]),
                    logical_error_rate=float(pt["logical_error_rate"]),
                    avg_syndrome_weight=float(pt.get("avg_syndrome_weight", 0.0)),
                    click_rate=float(pt.get("click_rate", 0.0)),
                )
            )
        sweeps.append(DistanceSweepResult(distance=int(sweep["distance"]), points=points))

    return ThresholdScenarioResult(
        name=str(data["name"]),
        init_label=str(data["init_label"]),
        track=str(data["track"]),
        distances=[int(d) for d in data["distances"]],
        sweeps=sweeps,
    )


def _x_axis(result: ThresholdScenarioResult, sweep: DistanceSweepResult) -> np.ndarray:
    # Same convention as src/simulation/code_threshold/threshold.py and plotting.py
    if result.track == "Z":
        return np.array([pt.p_x for pt in sweep.points], dtype=float)
    return np.array([pt.p_z for pt in sweep.points], dtype=float)


def _crossing_from_arrays(
    physical: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    min_logical_error_rate: float | None,
) -> float | None:
    diff = low - high
    for i in range(len(diff) - 1):
        if min_logical_error_rate is not None:
            rates = (low[i], high[i], low[i + 1], high[i + 1])
            if any(r < min_logical_error_rate for r in rates):
                continue
        if diff[i] == 0 or diff[i + 1] == 0:
            continue
        if diff[i] * diff[i + 1] < 0:
            p1, p2 = float(physical[i]), float(physical[i + 1])
            y1, y2 = float(diff[i]), float(diff[i + 1])
            return p1 - y1 * (p2 - p1) / (y2 - y1)
    return None


def _estimate_crossings(
    result: ThresholdScenarioResult,
    *,
    mode: CrossingMode,
    min_logical_error_rate: float | None,
) -> dict[tuple[int, int], float | None]:
    if len(result.sweeps) < 2:
        return {}

    sweeps = result.sweeps
    # Verify each sweep uses the same physical x-grid (required for meaningful crossings)
    x0 = _x_axis(result, sweeps[0])
    for sw in sweeps[1:]:
        x = _x_axis(result, sw)
        if len(x) != len(x0) or not np.allclose(x, x0, rtol=0, atol=0):
            raise ValueError("Physical grid differs between distances; cannot compute crossings reliably.")

    pairs: Iterable[tuple[int, int]]
    if mode == "neighbors":
        pairs = ((i, i + 1) for i in range(len(sweeps) - 1))
    else:
        pairs = ((i, j) for i in range(len(sweeps)) for j in range(i + 1, len(sweeps)))

    crossings: dict[tuple[int, int], float | None] = {}
    for i, j in pairs:
        sw_i, sw_j = sweeps[i], sweeps[j]
        low = np.array([pt.logical_error_rate for pt in sw_i.points], dtype=float)
        high = np.array([pt.logical_error_rate for pt in sw_j.points], dtype=float)
        crossings[(sw_i.distance, sw_j.distance)] = _crossing_from_arrays(
            x0, low, high, min_logical_error_rate
        )
    return crossings


def _pick_estimator(est: ThresholdEstimate, which: EstimatorName) -> float | None:
    if which == "best":
        return est.best_estimate
    if which == "weighted":
        return est.weighted_average
    if which == "simple":
        return est.simple_average
    raise AssertionError(which)


@dataclass(frozen=True)
class BootstrapReport:
    scenario: str
    crossings: CrossingMode
    estimator: EstimatorName
    shots: int
    B: int
    seed: int
    min_logical_error_rate: float | None
    point_estimate: float | None
    bootstrap_median: float | None
    bootstrap_mean: float | None
    bootstrap_std: float | None
    ci_level: float
    ci_low: float | None
    ci_high: float | None
    success_fraction: float


def _bootstrap_threshold(
    base: ThresholdScenarioResult,
    *,
    shots: int,
    B: int,
    seed: int,
    crossings_mode: CrossingMode,
    estimator: EstimatorName,
    min_logical_error_rate: float | None,
    ci_level: float,
) -> BootstrapReport:
    rng = np.random.default_rng(seed)

    base_crossings = _estimate_crossings(base, mode=crossings_mode, min_logical_error_rate=min_logical_error_rate)
    base_est = estimate_threshold(base_crossings)
    point_estimate = _pick_estimator(base_est, estimator)

    # Pre-extract p_hats by sweep to avoid repeated traversal overhead.
    x0 = _x_axis(base, base.sweeps[0])
    distances = [sw.distance for sw in base.sweeps]
    p_hats = [np.array([pt.logical_error_rate for pt in sw.points], dtype=float) for sw in base.sweeps]

    # Bootstrap distribution of the chosen estimator.
    values: list[float] = []
    failures = 0
    for _ in range(B):
        # Resample each distance curve pointwise: k' ~ Binomial(N, p_hat)
        p_boot = []
        for p in p_hats:
            p_clip = np.clip(p, 0.0, 1.0)
            k = rng.binomial(shots, p_clip)
            p_boot.append(k / shots)

        # Compute crossings for this replicate.
        crossings: dict[tuple[int, int], float | None] = {}
        if crossings_mode == "neighbors":
            pairs = [(i, i + 1) for i in range(len(distances) - 1)]
        else:
            pairs = [(i, j) for i in range(len(distances)) for j in range(i + 1, len(distances))]

        for i, j in pairs:
            crossings[(distances[i], distances[j])] = _crossing_from_arrays(
                x0, p_boot[i], p_boot[j], min_logical_error_rate
            )

        est = estimate_threshold(crossings)
        v = _pick_estimator(est, estimator)
        if v is None or not np.isfinite(v):
            failures += 1
            continue
        values.append(float(v))

    success_fraction = 0.0 if B == 0 else (len(values) / B)
    if not values:
        return BootstrapReport(
            scenario=base.name,
            crossings=crossings_mode,
            estimator=estimator,
            shots=shots,
            B=B,
            seed=seed,
            min_logical_error_rate=min_logical_error_rate,
            point_estimate=point_estimate,
            bootstrap_median=None,
            bootstrap_mean=None,
            bootstrap_std=None,
            ci_level=ci_level,
            ci_low=None,
            ci_high=None,
            success_fraction=success_fraction,
        )

    arr = np.array(values, dtype=float)
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(arr, [alpha, 1.0 - alpha])
    return BootstrapReport(
        scenario=base.name,
        crossings=crossings_mode,
        estimator=estimator,
        shots=shots,
        B=B,
        seed=seed,
        min_logical_error_rate=min_logical_error_rate,
        point_estimate=point_estimate,
        bootstrap_median=float(np.median(arr)),
        bootstrap_mean=float(np.mean(arr)),
        bootstrap_std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        ci_level=ci_level,
        ci_low=float(lo),
        ci_high=float(hi),
        success_fraction=success_fraction,
    )


def _format_float(x: float | None) -> str:
    if x is None:
        return "None"
    return f"{x:.6g}"


def main() -> None:
    args = _parse_args()
    
    # Resolve summary path: use preset, fall back to auto-detect if preset doesn't exist
    summary_path: Optional[Path] = PRESET_PATHS[args.preset]
    if summary_path is None or not summary_path.exists():
        # Fall back to first existing preset if the requested one doesn't exist
        summary_path = _default_summary_path()
    
    if summary_path is None or not summary_path.exists():
        preset_list = ", ".join(PRESET_PATHS.keys())
        raise SystemExit(
            f"Could not find threshold_summary.json for preset '{args.preset}'.\n"
            f"Available presets: {preset_list}\n"
            f"None of the preset files exist. Check that threshold experiments have been run."
        )

    summary = _load_json(summary_path)
    shots = int(summary["_metadata"]["shots"])

    min_lr = args.min_logical_error_rate
    if min_lr is None:
        min_lr = 5.0 / shots

    scenarios: dict[str, Any] = dict(summary["scenarios"])
    if args.scenario is not None:
        if args.scenario not in scenarios:
            raise SystemExit(f"Scenario '{args.scenario}' not found in summary.")
        scenarios = {args.scenario: scenarios[args.scenario]}

    reports: list[BootstrapReport] = []
    for name, entry in scenarios.items():
        scenario_json = Path(entry["json"])
        # Summary stores relative-to-project paths
        if not scenario_json.is_absolute():
            scenario_json = PROJECT_ROOT / scenario_json
        data = _load_json(scenario_json)
        result = _scenario_result_from_json(data)

        rep = _bootstrap_threshold(
            result,
            shots=shots,
            B=int(args.B),
            seed=int(args.seed),
            crossings_mode=args.crossings,
            estimator=args.estimator,
            min_logical_error_rate=float(min_lr) if min_lr is not None else None,
            ci_level=float(args.ci),
        )
        reports.append(rep)

    if args.json_out is not None:
        out = [r.__dict__ for r in reports]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as fh:
            json.dump(out, fh, indent=2)
        print(f"Wrote {len(reports)} report(s) to {args.json_out}")
        return

    # Print a compact table
    print(f"summary: {summary_path}")
    print(f"shots: {shots}  B: {args.B}  seed: {args.seed}  crossings: {args.crossings}  estimator: {args.estimator}")
    print(f"min_logical_error_rate: {min_lr}  CI: {args.ci}")
    print()
    header = (
        "scenario",
        "point_est",
        "boot_median",
        "ci_low",
        "ci_high",
        "boot_std",
        "success",
    )
    print("{:18s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>8s}".format(*header))
    for r in reports:
        print(
            "{:18s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>8.1%}".format(
                r.scenario[:18],
                _format_float(r.point_estimate),
                _format_float(r.bootstrap_median),
                _format_float(r.ci_low),
                _format_float(r.ci_high),
                _format_float(r.bootstrap_std),
                r.success_fraction,
            )
        )


if __name__ == "__main__":
    main()


