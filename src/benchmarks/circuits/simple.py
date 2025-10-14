"""Simple single-qubit logical benchmark: only H, X, Z."""

from qiskit.circuit import QuantumCircuit

from ..BenchmarkCircuit import BenchmarkCircuit


class Simple1QXZHBenchmark(BenchmarkCircuit):
    """
    A minimal 1-qubit circuit containing only single-qubit logical gates {H, X, Z}.

    The default sequence is H → X → Z on a single logical qubit, suitable for
    driving the logical frame and end-basis calculation in the surface-code
    simulation pipeline.
    """

    def build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(1, name="simple_1q_xzh")
        qc.h(0)
        #qc.h(0)
        qc.z(0)
        #qc.x(0)
        return qc
