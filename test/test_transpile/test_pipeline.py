"""Integration tests for :class:`transpile.pipeline.HeavyHexTranspiler` helpers.

Included checks:
- ``test_select_best_orders_candidates`` verifies leaderboard ordering by metrics.
- ``test_run_baseline_selects_lowest_twoq`` monkeypatches pass steps to assert the selector prefers low two-qubit counts.
- ``test_run_qec_round_delegates_to_baseline`` confirms the wrapper reuses ``run_baseline``.
"""

import pytest

from qiskit import QuantumCircuit

from transpile.config import TranspileConfig
from transpile.pipeline import HeavyHexTranspiler
from transpile import steps


def test_select_best_orders_candidates(simple_target):
    qc = QuantumCircuit(1)
    candidates = [
        (qc, {"twoq": 2, "depth": 5, "duration_ns": 5.0}),
        (qc, {"twoq": 1, "depth": 7, "duration_ns": 1.0}),
        (qc, {"twoq": 1, "depth": 4, "duration_ns": 2.0}),
    ]

    best_qc, best_metrics, leaderboard = HeavyHexTranspiler._select_best(candidates, top_k=2)
    assert best_metrics["twoq"] == 1
    assert best_metrics["depth"] == 4
    assert len(leaderboard) == 2
    assert leaderboard[0][1]["depth"] <= leaderboard[1][1]["depth"]


def test_run_baseline_selects_lowest_twoq(monkeypatch, simple_target):
    cfg = TranspileConfig(target=simple_target, seeds=3, seed_offset=3, keep_top_k=2)
    transpiler = HeavyHexTranspiler(cfg)

    base_qc = QuantumCircuit(3)
    base_qc.cx(0, 1)

    def clone_with_seed(circ, seed=None):
        new_circ = circ.copy()
        new_circ._seed = seed if seed is not None else getattr(circ, "_seed", None)
        return new_circ

    monkeypatch.setattr(steps, "unroll", lambda qc, target, basis: clone_with_seed(qc))
    monkeypatch.setattr(steps, "initial_layout", lambda qc, target, seed, max_iterations=5: clone_with_seed(qc, seed))
    monkeypatch.setattr(steps, "route", lambda qc, target, seed: clone_with_seed(qc))
    monkeypatch.setattr(steps, "gate_direction", lambda qc, target: clone_with_seed(qc))
    monkeypatch.setattr(steps, "opt_local", lambda qc: clone_with_seed(qc))
    monkeypatch.setattr(steps, "schedule", lambda qc, target, mode, dd_policy=None: clone_with_seed(qc))

    def fake_score(qc, _target):
        seed = getattr(qc, "_seed", 99)
        return {
            "twoq": int(seed),
            "depth": 20 - int(seed),
            "duration_ns": float(seed),
            "n_qubits": qc.num_qubits,
            "swaps": 0,
            "dir_fixes": 0.0,
        }

    monkeypatch.setattr(steps, "score", fake_score)

    best_qc, metrics, leaderboard = transpiler.run_baseline(base_qc)

    assert metrics["twoq"] == 3
    assert len(leaderboard) == cfg.keep_top_k
    assert [entry[1]["twoq"] for entry in leaderboard] == [3, 4]


def test_run_qec_round_delegates_to_baseline(monkeypatch, simple_target):
    cfg = TranspileConfig(target=simple_target, seeds=1)
    transpiler = HeavyHexTranspiler(cfg)

    called = {}

    def fake_run_baseline(self, qc):
        called["flag"] = True
        return qc, {"twoq": 0}, [(qc, {"twoq": 0})]

    monkeypatch.setattr(HeavyHexTranspiler, "run_baseline", fake_run_baseline)

    qc = QuantumCircuit(2)
    result = transpiler.run_qec_round(qc)

    assert called["flag"] is True
    assert result[1]["twoq"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vv", "-rA"]))
