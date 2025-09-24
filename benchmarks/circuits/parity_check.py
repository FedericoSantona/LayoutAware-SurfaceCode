"""Parity-check (syndrome toy) logical benchmark circuit."""

from qiskit.circuit import QuantumCircuit

from ..BenchmarkCircuit import BenchmarkCircuit


class ParityCheckBenchmark(BenchmarkCircuit):
    """Measure the parity of three data qubits onto an ancilla."""

    def build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(4, 1, name="parity_check")

        # Accumulate the parity of data qubits (0, 1, 2) onto ancilla qubit 3
        qc.cx(0, 3)
        qc.cx(1, 3)
        qc.cx(2, 3)
        qc.measure(3, 0)
        return qc
