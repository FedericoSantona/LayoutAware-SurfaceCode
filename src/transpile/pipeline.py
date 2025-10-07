from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from qiskit import QuantumCircuit

from .config import TranspileConfig
from . import steps


class HeavyHexTranspiler:
    """
    Thin façade that orchestrates the pure steps with multi-seed exploration.
    Produces:
      - best candidate circuit,
      - its metrics,
      - a leaderboard of the top-k (circuit, metrics) pairs.
    """

    def __init__(self, cfg: TranspileConfig):
        self.cfg = cfg

    # --------------------------- Public entry points ---------------------------

    def run_baseline(
        self, qc: QuantumCircuit
    ) -> Tuple[QuantumCircuit, Dict[str, Any], List[Tuple[QuantumCircuit, Dict[str, Any]]]]:
        """
        Transpile a logical (pre-QEC) circuit to heavy-hex with layout, routing, scheduling.
        """
        q0 = steps.unroll(qc, self.cfg.target, self.cfg.basis)

        candidates: List[Tuple[QuantumCircuit, Dict[str, Any]]] = []
        for seed in self.cfg.seed_stream():
            q1 = steps.initial_layout(q0, self.cfg.target, seed, max_iterations=self.cfg.sabre_layout_iterations)
            q2 = steps.route(q1, self.cfg.target, seed)
            if self.cfg.enable_gate_direction_fix:
                q2 = steps.gate_direction(q2, self.cfg.target)
            q3 = steps.opt_local(q2)
            q4 = steps.schedule(q3, self.cfg.target, self.cfg.schedule_mode, self.cfg.dd_policy)
            metrics = steps.score(q4, self.cfg.target)
            candidates.append((q4, metrics))

        best, best_metrics, leaderboard = self._select_best(candidates, self.cfg.keep_top_k)
        return best, best_metrics, leaderboard

    def run_qec_round(
        self, qc_round: QuantumCircuit
    ) -> Tuple[QuantumCircuit, Dict[str, Any], List[Tuple[QuantumCircuit, Dict[str, Any]]]]:
        """
        Transpile a *single QEC round template* (A/B/C/D barriers already present).
        The flow mirrors run_baseline.
        """
        return self.run_baseline(qc_round)

    # --------------------------- Internal helpers -----------------------------

    @staticmethod
    def _select_best(
        cands: List[Tuple[QuantumCircuit, Dict[str, Any]]], top_k: int
    ) -> Tuple[QuantumCircuit, Dict[str, Any], List[Tuple[QuantumCircuit, Dict[str, Any]]]]:
        """
        Order candidates by (twoq, depth, duration_ns) and return best + top-k leaderboard.
        """
        def key(item):
            _, m = item
            return (
                int(m.get("twoq", 1 << 30)),
                int(m.get("depth", 1 << 30)),
                float(m.get("duration_ns") if m.get("duration_ns") is not None else 1e99),
            )

        ordered = sorted(cands, key=key)
        best_qc, best_metrics = ordered[0]
        leaderboard = ordered[: max(1, int(top_k))]
        return best_qc, best_metrics, leaderboard