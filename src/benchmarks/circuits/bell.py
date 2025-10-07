"""Bell-state logical benchmark circuit."""

from qiskit.circuit import QuantumCircuit

from ..BenchmarkCircuit import BenchmarkCircuit


class BellStateBenchmark(BenchmarkCircuit):
    """Prepare a logical Bell state using two qubits and a single CX."""

    def build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2, name="bell_state")
        qc.h(0)
        qc.cx(0, 1)
        return qc
