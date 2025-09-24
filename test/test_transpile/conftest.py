"""Shared pytest fixtures for the transpile test suite.

Provides a minimal heavy-hex-like Qiskit ``Target`` fixture used by config,
steps, and pipeline tests to avoid duplicating setup logic.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

qiskit = pytest.importorskip("qiskit")

from qiskit.circuit import Parameter
from qiskit.circuit.library import CXGate, IGate, RZGate, SXGate, XGate, YGate
from qiskit.transpiler import InstructionProperties, Target


def _single_qubit_props(num_qubits: int, duration: float = 1.0):
    return {
        (idx,): InstructionProperties(duration=duration, error=0.0)
        for idx in range(num_qubits)
    }


@pytest.fixture
def simple_target():
    target = Target(num_qubits=3, dt=1e-9)

    rz_theta = Parameter("theta")
    basis_gates = (
        RZGate(rz_theta),
        SXGate(),
        XGate(),
        YGate(),
        IGate(),
    )
    for gate in basis_gates:
        target.add_instruction(gate, _single_qubit_props(target.num_qubits, duration=1.0))

    cx_props = {
        (0, 1): InstructionProperties(duration=2.0, error=0.0),
        (1, 2): InstructionProperties(duration=2.0, error=0.0),
    }
    target.add_instruction(CXGate(), cx_props)
    return target
