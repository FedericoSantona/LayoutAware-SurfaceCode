"""Structure checks for the benchmark circuit subclasses.

Covered scenarios:
- ``test_bell_state_benchmark_structure`` validates qubit count and gate makeup of the Bell circuit.
- ``test_ghz3_benchmark_structure`` inspects GHZ-3 entanglement layout and two-qubit usage.
- ``test_teleportation_benchmark_structure`` ensures teleportation template includes measurements and corrections.
- ``test_parity_check_benchmark_structure`` confirms parity-check ancilla logic and measurement count.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

qiskit = pytest.importorskip("qiskit")

from benchmarks import (
    BellStateBenchmark,
    GHZ3Benchmark,
    ParityCheckBenchmark,
    TeleportationBenchmark,
)


def _count_twoq_ops(qc):
    ops = qc.count_ops()
    return sum(ops.get(name, 0) for name in ("cx", "cz", "swap", "rxx", "ryy", "rzz"))


def test_bell_state_benchmark_structure():
    bench = BellStateBenchmark()
    qc = bench.get_circuit()

    assert qc.num_qubits == 2
    assert qc.name == "bell_state"
    assert _count_twoq_ops(qc) == 1
    assert qc.count_ops().get("h", 0) == 1


def test_ghz3_benchmark_structure():
    bench = GHZ3Benchmark()
    qc = bench.get_circuit()

    assert qc.num_qubits == 3
    assert qc.name == "ghz_3"
    assert _count_twoq_ops(qc) == 2
    assert qc.count_ops().get("h", 0) == 1


def test_teleportation_benchmark_structure():
    bench = TeleportationBenchmark()
    qc = bench.get_circuit()

    assert qc.num_qubits == 3
    assert qc.name == "quantum_teleportation"
    assert _count_twoq_ops(qc) == 2
    assert qc.count_ops().get("measure", 0) == 2


def test_parity_check_benchmark_structure():
    bench = ParityCheckBenchmark()
    qc = bench.get_circuit()

    assert qc.num_qubits == 4
    assert qc.name == "parity_check"
    assert _count_twoq_ops(qc) == 3
    assert qc.count_ops().get("measure", 0) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vv", "-rA"]))
