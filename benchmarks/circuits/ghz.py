"""GHZ(3) logical benchmark circuit."""

from qiskit.circuit import QuantumCircuit

from ..BenchmarkCircuit import BenchmarkCircuit


class GHZ3Benchmark(BenchmarkCircuit):
    """Prepare a 3-qubit GHZ state with two CX layers."""

    def build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(3, name="ghz_3")
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        return qc
