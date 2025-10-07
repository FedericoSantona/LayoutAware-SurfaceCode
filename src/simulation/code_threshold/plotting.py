"""Plotting helpers for threshold sweeps."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np

from .threshold import DistanceSweepResult, ThresholdScenarioResult


def _x_values(result: ThresholdScenarioResult, sweep: DistanceSweepResult) -> np.ndarray:
    if result.track == "Z":
        return np.array([pt.p_x for pt in sweep.points])
    return np.array([pt.p_z for pt in sweep.points])


def plot_scenario(result: ThresholdScenarioResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for sweep in result.sweeps:
        x_vals = _x_values(result, sweep)
        y_vals = np.array([pt.logical_error_rate for pt in sweep.points])
        ax.plot(x_vals, y_vals, marker='o', label=f"d={sweep.distance}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    axis_label = "p_X" if result.track == "Z" else ("p_Z" if result.track == "X" else "p")
    ax.set_xlabel(f"Physical error rate ({axis_label})")
    ax.set_ylabel("Logical error rate")
    ax.set_title(result.name.replace('_', ' '))
    ax.grid(True, which='both', ls=':')
    ax.legend()
    out_path = output_dir / f"{result.name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def export_csv(result: ThresholdScenarioResult, output_dir: Path) -> Dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[int, Path] = {}
    for sweep in result.sweeps:
        path = output_dir / f"{result.name}_d{sweep.distance}.csv"
        with path.open('w') as fh:
            fh.write("p_x,p_z,logical_error_rate,avg_syndrome_weight,click_rate\n")
            for point in sweep.points:
                fh.write(
                    f"{point.p_x:.6e},{point.p_z:.6e},{point.logical_error_rate:.6e},{point.avg_syndrome_weight:.6e},{point.click_rate:.6e}\n"
                )
        paths[sweep.distance] = path
    return paths
