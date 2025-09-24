"""Validation of granular transpilation steps in :mod:`transpile.steps`.

Coverage breakdown:
- ``test_allowed_cx_directions`` inspects target-derived CX direction map.
- ``test_count_2q_and_swaps`` validates logical two-qubit and swap counting helper.
- ``test_estimate_dir_fix_fraction*`` ensures direction-violation heuristic behaves with/without CX gates.
- ``test_make_dd_sequence_*`` exercises DD sequence construction and error handling.
- ``test_unroll_converts_to_basis`` verifies BasisTranslator reduces outside-basis gates.
- ``test_transpile_step_chain`` runs the full pass pipeline and checks metrics.
- ``test_schedule_*`` covers scheduling modes and invalid option guard.
- ``test_score_reports_swap_and_dirfix`` inspects metric reporting for routed circuits.
"""

import pytest

from qiskit import QuantumCircuit

from transpile import steps


def test_allowed_cx_directions(simple_target):
    allowed = steps._allowed_cx_directions(simple_target)
    assert allowed == {(0, 1), (1, 2)}


def test_count_2q_and_swaps():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.swap(1, 2)
    qc.cx(2, 1)
    twoq, swaps = steps._count_2q_and_swaps(qc)
    assert twoq == 3
    assert swaps == 1


def test_estimate_dir_fix_fraction(simple_target):
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(2, 1)
    fraction = steps._estimate_dir_fix_fraction(qc, simple_target)
    assert pytest.approx(fraction) == 0.5


def test_estimate_dir_fix_fraction_no_cx(simple_target):
    qc = QuantumCircuit(3)
    assert steps._estimate_dir_fix_fraction(qc, simple_target) == 0.0


def test_make_dd_sequence_variants():
    seq_xix = steps._make_dd_sequence("XIX")
    seq_xyxy = steps._make_dd_sequence("XYXY")
    assert [gate.name for gate in seq_xix] == ["x", "id", "x", "id"]
    assert [gate.name for gate in seq_xyxy] == ["x", "y", "x", "y"]


def test_make_dd_sequence_invalid_policy():
    with pytest.raises(ValueError):
        steps._make_dd_sequence("ZZZZ")


def test_unroll_converts_to_basis(simple_target):
    qc = QuantumCircuit(1)
    qc.h(0)
    out = steps.unroll(qc, simple_target, basis=("rz", "sx", "x", "cx"))
    assert "h" not in out.count_ops()


def test_transpile_step_chain(simple_target):
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)

    q0 = steps.unroll(qc, simple_target, basis=("rz", "sx", "x", "cx"))
    q1 = steps.initial_layout(q0, simple_target, seed=7)
    q2 = steps.route(q1, simple_target, seed=7)
    q3 = steps.gate_direction(q2, simple_target)
    q4 = steps.opt_local(q3)
    q5 = steps.schedule(q4, simple_target, mode="alap")

    metrics = steps.score(q5, simple_target)
    assert metrics["n_qubits"] == qc.num_qubits
    assert "duration_ns" in metrics


def test_schedule_mode_variants(simple_target):
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    steps.schedule(qc, simple_target, mode="asap")
    steps.schedule(qc, simple_target, mode="alap", dd_policy="XIX")


def test_schedule_invalid_mode(simple_target):
    qc = QuantumCircuit(1)
    with pytest.raises(ValueError):
        steps.schedule(qc, simple_target, mode="latest")


def test_score_reports_swap_and_dirfix(simple_target):
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.swap(1, 2)
    qc.cx(2, 1)
    metrics = steps.score(qc, simple_target)

    assert metrics["twoq"] == 3
    assert metrics["swaps"] == 1
    assert pytest.approx(metrics["dir_fixes"], rel=1e-3) == 0.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vv", "-rA"]))
