"""Unit tests for :mod:`benchmarks.BenchmarkCircuit` helpers.

Test matrix:
- ``test_get_circuit_is_cached`` ensures subclass circuits are constructed once and cached.
- ``test_get_circuit_type_validation`` verifies invalid subclass returns raise ``TypeError``.
- ``test_compute_logical_metrics`` checks logical metric counting against circuit structure.
- ``test_to_qasm_contains_openqasm_header`` validates OpenQASM export path.
- ``test_to_yaml_round_trip`` confirms YAML serialization schema and content.
- ``test_param_to_serializable_*`` families cover parameter serialization helpers.
- ``test_benchmark_library_metrics`` exercises the public daughter classes and their metrics.
"""

import os
import sys

import pytest

# Ensure the repository root is on sys.path so local packages are importable
# when running this test module directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Skip the module if its heavy dependencies are missing in the environment.
qiskit = pytest.importorskip("qiskit")
yaml = pytest.importorskip("yaml")

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from benchmarks.BenchmarkCircuit import BenchmarkCircuit, _param_to_serializable
from benchmarks import (
    BellStateBenchmark,
    GHZ3Benchmark,
    ParityCheckBenchmark,
    TeleportationBenchmark,
)


class DummyBenchmark(BenchmarkCircuit):
    """Concrete BenchmarkCircuit for exercising the base-class helpers."""

    def __init__(self) -> None:
        super().__init__()
        self.build_calls = 0

    def build_circuit(self) -> QuantumCircuit:  # type: ignore[override]
        self.build_calls += 1
        qc = QuantumCircuit(2, 2, name="dummy")
        qc.h(0)
        qc.cx(0, 1)
        qc.swap(0, 1)
        qc.rxx(0.2, 0, 1)
        qc.measure([0, 1], [0, 1])
        return qc


class BadBenchmark(BenchmarkCircuit):
    def build_circuit(self):  # type: ignore[override]
        return "not a circuit"


@pytest.fixture
def dummy_benchmark() -> DummyBenchmark:
    return DummyBenchmark()


def test_get_circuit_is_cached(dummy_benchmark: DummyBenchmark) -> None:
    first = dummy_benchmark.get_circuit()
    second = dummy_benchmark.get_circuit()

    assert first is second
    assert dummy_benchmark.build_calls == 1


def test_get_circuit_type_validation() -> None:
    bad = BadBenchmark()
    with pytest.raises(TypeError):
        bad.get_circuit()


def test_compute_logical_metrics(dummy_benchmark: DummyBenchmark) -> None:
    metrics = dummy_benchmark.compute_logical_metrics()
    qc = dummy_benchmark.get_circuit()

    assert metrics["n_qubits"] == qc.num_qubits
    assert metrics["depth"] == qc.depth()
    assert metrics["twoq"] == 3  # cx + swap + rxx


def test_to_qasm_contains_openqasm_header(dummy_benchmark: DummyBenchmark) -> None:
    qasm = dummy_benchmark.to_qasm()

    assert "OPENQASM" in qasm
    assert "cx" in qasm


def test_to_yaml_round_trip(dummy_benchmark: DummyBenchmark) -> None:
    yaml_str = dummy_benchmark.to_yaml()
    payload = yaml.safe_load(yaml_str)

    assert payload["version"] == 1
    assert payload["name"] == "dummy"
    assert payload["qubits"] == 2
    assert payload["clbits"] == 2
    assert any(step["name"] == "swap" for step in payload["instructions"])


def test_param_to_serializable_numeric_coercion() -> None:
    assert _param_to_serializable(2.0) == 2
    assert _param_to_serializable(0.5) == pytest.approx(0.5)


def test_param_to_serializable_parameter_expression() -> None:
    theta = Parameter("theta")
    assert _param_to_serializable(theta) == "theta"


@pytest.mark.parametrize(
    "benchmark_cls, expected",
    [
        (BellStateBenchmark, {"name": "bell_state", "qubits": 2, "twoq": 1}),
        (GHZ3Benchmark, {"name": "ghz_3", "qubits": 3, "twoq": 2}),
        (TeleportationBenchmark, {"name": "quantum_teleportation", "qubits": 3, "twoq": 2}),
        (ParityCheckBenchmark, {"name": "parity_check", "qubits": 4, "twoq": 3}),
    ],
)
def test_benchmark_library_metrics(benchmark_cls, expected) -> None:
    bench = benchmark_cls()
    qc = bench.get_circuit()
    metrics = bench.compute_logical_metrics()

    assert qc.name == expected["name"]
    assert qc.num_qubits == expected["qubits"]
    assert metrics["n_qubits"] == expected["qubits"]
    assert metrics["twoq"] == expected["twoq"]


if __name__ == "__main__":
    # Allow running this module directly for a quick, verbose smoke check.
    raise SystemExit(pytest.main([__file__, "-vv", "-rA"]))
