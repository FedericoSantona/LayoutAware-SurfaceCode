"""Quantum teleportation logical benchmark circuit."""

from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import XGate, ZGate

from ..BenchmarkCircuit import BenchmarkCircuit


class TeleportationBenchmark(BenchmarkCircuit):
    """Teleport an unknown state from qubit 0 to qubit 2 using two CX gates."""

    def build_circuit(self) -> QuantumCircuit:
        c0 = ClassicalRegister(1, "m0")
        c1 = ClassicalRegister(1, "m1")
        qc = QuantumCircuit(3, name="quantum_teleportation")
        qc.add_register(c0)
        qc.add_register(c1)

        # EPR pair shared between qubits 1 and 2
        qc.h(1)
        qc.cx(1, 2)

        # Bell-state measurement between qubits 0 and 1
        qc.cx(0, 1)
        qc.h(0)
        qc.measure(0, c0[0])
        qc.measure(1, c1[0])

        # Conditional corrections on the destination qubit (post-measurement)
        z_cond = ZGate().to_mutable()
        z_cond.condition = (c0, 1)
        qc.append(z_cond, [2])

        x_cond = XGate().to_mutable()
        x_cond.condition = (c1, 1)
        qc.append(x_cond, [2])

        return qc
