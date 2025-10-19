"""Comprehensive Pauli tracking and frame management.

This module provides:
- Low-level symplectic Pauli conjugation engine (PauliOperator, conjugate_circuit)
- High-level Pauli frame management (PauliFrameManager)
- Single source of truth for all Pauli tracking logic

It implements a symplectic Pauli tracker that computes U†σU for demo operators,
where U is the logical circuit. This makes demos reflect the "as-if physical" state
after virtual gates are applied in the Pauli frame.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from qiskit.circuit import QuantumCircuit


class PauliOperator:
    """Represents a Pauli operator as symplectic tableau (x, z) bitsets.
    
    For n logical qubits, represents a Pauli product by two length-n bitsets:
    - x[i] = 1 means X acts on logical qubit i
    - z[i] = 1 means Z acts on logical qubit i
    """
    
    def __init__(self, n_qubits: int, x_bits: int = 0, z_bits: int = 0, phase: int = +1):
        """Initialize Pauli operator with n qubits.
        
        Args:
            n_qubits: Number of logical qubits
            x_bits: Bitmask for X operators (bit i = 1 if X acts on qubit i)
            z_bits: Bitmask for Z operators (bit i = 1 if Z acts on qubit i)
        """
        self.n_qubits = n_qubits
        self.x_bits = x_bits & ((1 << n_qubits) - 1)  # Mask to n_qubits bits
        self.z_bits = z_bits & ((1 << n_qubits) - 1)  # Mask to n_qubits bits
        self._phase = +1 if phase >= 0 else -1
    
    @classmethod
    def single_qubit_x(cls, n_qubits: int, qubit: int) -> 'PauliOperator':
        """Create single-qubit X operator on specified qubit."""
        if qubit >= n_qubits:
            raise ValueError(f"Qubit {qubit} out of range for {n_qubits} qubits")
        return cls(n_qubits, x_bits=1 << qubit, z_bits=0, phase=+1)
    
    @classmethod
    def single_qubit_z(cls, n_qubits: int, qubit: int) -> 'PauliOperator':
        """Create single-qubit Z operator on specified qubit."""
        if qubit >= n_qubits:
            raise ValueError(f"Qubit {qubit} out of range for {n_qubits} qubits")
        return cls(n_qubits, x_bits=0, z_bits=1 << qubit, phase=+1)
    
    def conjugate_H(self, qubit: int) -> None:
        """Apply H conjugation: swap x[qubit] ↔ z[qubit]."""
        if qubit >= self.n_qubits:
            raise ValueError(f"Qubit {qubit} out of range for {self.n_qubits} qubits")
        
        # Swap x and z bits for this qubit
        x_bit = (self.x_bits >> qubit) & 1
        z_bit = (self.z_bits >> qubit) & 1
        
        # Clear both bits
        self.x_bits &= ~(1 << qubit)
        self.z_bits &= ~(1 << qubit)
        
        # Set swapped bits
        self.x_bits |= z_bit << qubit
        self.z_bits |= x_bit << qubit
    
    def conjugate_CNOT(self, control: int, target: int) -> None:
        """Apply CNOT conjugation: x[target] ^= x[control], z[control] ^= z[target]."""
        if control >= self.n_qubits or target >= self.n_qubits:
            raise ValueError(f"Qubit indices out of range for {self.n_qubits} qubits")
        
        # x[target] ^= x[control]
        control_x = (self.x_bits >> control) & 1
        self.x_bits ^= control_x << target
        
        # z[control] ^= z[target]
        target_z = (self.z_bits >> target) & 1
        self.z_bits ^= target_z << control
    
    def conjugate_X(self, qubit: int) -> None:
        """Apply X conjugation: toggle phase if operator anticommutes with X on qubit.

        X O X = O if O ∈ {I, X} on this qubit, and = -O if O ∈ {Y, Z} on this qubit.
        Structure is unchanged for single-qubit rotations in this representation.
        """
        if qubit >= self.n_qubits:
            raise ValueError(f"Qubit {qubit} out of range for {self.n_qubits} qubits")
        # Anticommutes iff Z acts on this qubit (including Y which has both X and Z)
        z_bit = (self.z_bits >> qubit) & 1
        if z_bit:
            self._phase *= -1
    
    def conjugate_Z(self, qubit: int) -> None:
        """Apply Z conjugation: toggle phase if operator anticommutes with Z on qubit.

        Z O Z = O if O ∈ {I, Z} on this qubit, and = -O if O ∈ {X, Y} on this qubit.
        Structure is unchanged for single-qubit rotations in this representation.
        """
        if qubit >= self.n_qubits:
            raise ValueError(f"Qubit {qubit} out of range for {self.n_qubits} qubits")
        # Anticommutes iff X acts on this qubit (including Y)
        x_bit = (self.x_bits >> qubit) & 1
        if x_bit:
            self._phase *= -1
    
    def to_string(self) -> str:
        """Return symbolic string representation like 'X(q0)*Z(q1)'."""
        terms = []
        
        for i in range(self.n_qubits):
            x_bit = (self.x_bits >> i) & 1
            z_bit = (self.z_bits >> i) & 1
            
            if x_bit and z_bit:
                terms.append(f"Y(q{i})")
            elif x_bit:
                terms.append(f"X(q{i})")
            elif z_bit:
                terms.append(f"Z(q{i})")
        
        if not terms:
            return "I"
        
        return "*".join(terms)
    
    def get_qubit_pauli(self, qubit: int) -> str:
        """Get the Pauli operator acting on a specific qubit."""
        if qubit >= self.n_qubits:
            raise ValueError(f"Qubit {qubit} out of range for {self.n_qubits} qubits")
        
        x_bit = (self.x_bits >> qubit) & 1
        z_bit = (self.z_bits >> qubit) & 1
        
        if x_bit and z_bit:
            return "Y"
        elif x_bit:
            return "X"
        elif z_bit:
            return "Z"
        else:
            return "I"
    
    @classmethod
    def two_qubit_zz(cls, n_qubits: int, q0: int, q1: int) -> 'PauliOperator':
        """Create Z⊗Z operator on two qubits."""
        if q0 >= n_qubits or q1 >= n_qubits:
            raise ValueError(f"Qubit indices out of range for {n_qubits} qubits")
        z_bits = (1 << q0) | (1 << q1)
        return cls(n_qubits, x_bits=0, z_bits=z_bits)
    
    @classmethod
    def two_qubit_xx(cls, n_qubits: int, q0: int, q1: int) -> 'PauliOperator':
        """Create X⊗X operator on two qubits."""
        if q0 >= n_qubits or q1 >= n_qubits:
            raise ValueError(f"Qubit indices out of range for {n_qubits} qubits")
        x_bits = (1 << q0) | (1 << q1)
        return cls(n_qubits, x_bits=x_bits, z_bits=0)
    
    def copy(self) -> 'PauliOperator':
        """Return a copy of this Pauli operator."""
        return PauliOperator(self.n_qubits, self.x_bits, self.z_bits, self._phase)

    def phase_sign(self) -> int:
        """Return the global phase sign (±1) accumulated during conjugations."""
        return +1 if self._phase >= 0 else -1


def conjugate_circuit(initial_pauli: PauliOperator, qiskit_circuit: QuantumCircuit) -> PauliOperator:
    """Walk circuit backwards applying conjugations to compute U†σU.
    
    Args:
        initial_pauli: Starting Pauli operator (e.g., Z on qubit 0)
        qiskit_circuit: The logical circuit to conjugate through
        
    Returns:
        Final PauliOperator after conjugation
    """
    pauli = initial_pauli.copy()
    
    # Walk circuit backwards (reverse order)
    for instruction in reversed(qiskit_circuit.data):
        gate_name = instruction.operation.name.lower()
        qubits = [qiskit_circuit.find_bit(qb).index for qb in instruction.qubits]
        
        if gate_name == "h":
            if len(qubits) != 1:
                raise ValueError("H gate must act on single qubit")
            pauli.conjugate_H(qubits[0])
        
        elif gate_name in {"cx", "cnot", "cz"}:
            if len(qubits) != 2:
                raise ValueError("CNOT gate must act on two qubits")
            control, target = qubits[0], qubits[1]
            pauli.conjugate_CNOT(control, target)
        
        elif gate_name == "x":
            if len(qubits) != 1:
                raise ValueError("X gate must act on single qubit")
            pauli.conjugate_X(qubits[0])
        
        elif gate_name == "z":
            if len(qubits) != 1:
                raise ValueError("Z gate must act on single qubit")
            pauli.conjugate_Z(qubits[0])
        
        # Ignore other gates (barrier, measure, etc.)
    
    return pauli


class PauliFrameManager:
    """Manages Pauli frame state and virtual gate tracking for logical qubits.

    frame[qname]["fx"] is the bit used to flip Z-basis singles (or joint ZZ via XOR),
    frame[qname]["fz"] is the bit used to flip X-basis singles (or joint XX via XOR).
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = int(n_qubits)
        self.frame: Dict[str, Dict[str, Any]] = {f"q{i}": {"fx": 0, "fz": 0} for i in range(self.n_qubits)}
        self.virtual_gates: Dict[str, List[str]] = {f"q{i}": [] for i in range(self.n_qubits)}

    # ---------- Virtual gates ----------
    def add_virtual_gate(self, qubit_index: int, gate: str) -> None:
        """Add virtual gate and update frame bits."""
        qname = f"q{int(qubit_index)}"
        self.virtual_gates[qname].append(gate.upper())
        self._update_frame_from_virtual_gates(qubit_index)

    def _update_frame_from_virtual_gates(self, qubit_index: int) -> None:
        """Update frame bits based on virtual gate sequence."""
        qname = f"q{int(qubit_index)}"
        seq = self.virtual_gates[qname]
        # Use the comprehensive conjugation from pauli_tracker
        _, z_phase = self._conjugate_axis_and_phase("Z", seq)
        _, x_phase = self._conjugate_axis_and_phase("X", seq)
        self.frame[qname]["fx"] = (1 if z_phase < 0 else 0)
        self.frame[qname]["fz"] = (1 if x_phase < 0 else 0)

    @staticmethod
    def _conjugate_axis_and_phase(axis: str, gates: List[str]) -> Tuple[str, int]:
        """Heisenberg-conjugate using the comprehensive engine."""
        # Create a temporary PauliOperator and conjugate through gates
        n_qubits = 1  # Single qubit for virtual gates
        if axis.upper() == "Z":
            pauli = PauliOperator.single_qubit_z(n_qubits, 0)
        else:
            pauli = PauliOperator.single_qubit_x(n_qubits, 0)
        
        # Apply gates in reverse order (right-to-left)
        for gate in reversed(gates):
            gate = gate.upper()
            if gate == "H":
                pauli.conjugate_H(0)
            elif gate == "X":
                pauli.conjugate_X(0)
            elif gate == "Z":
                pauli.conjugate_Z(0)
        
        axis_result = pauli.get_qubit_pauli(0)
        phase_result = pauli.phase_sign()
        return axis_result, phase_result

    # ---------- CNOT parity updates ----------
    def apply_cnot_update(self, control: str, target: str, m_zz: np.ndarray, m_xx: np.ndarray) -> None:
        """Update Pauli frame given CNOT parity bits.

        Contract: fz[target] ^= m_ZZ, fx[control] ^= m_XX.
        """
        self.frame[target]["fz"] ^= m_zz
        self.frame[control]["fx"] ^= m_xx

    # ---------- Queries ----------
    def get_frame_bit(self, qubit_name: str, basis_axis: str) -> int:
        """Get frame bit for a qubit and basis."""
        key = "fx" if basis_axis.upper() == "Z" else "fz"
        v = self.frame.get(qubit_name, {}).get(key, 0)
        if isinstance(v, np.ndarray):
            return int(round(float(v.mean()))) & 1
        return int(v) & 1

    # ---------- Final operator computation ----------
    def get_final_operator_info(
        self,
        qubit_index: int,
        initial_basis: str,
        qiskit_circuit: QuantumCircuit,
    ) -> Dict[str, Any]:
        """Return final operator info for single-qubit demo/snapshot.

        initial_basis: 'Z' or 'X' (requested basis for this single-qubit op)
        Returns: dict with keys: axis ('Z'|'X'), phase (±1), operator_string, pauli_operator
        """
        n = qiskit_circuit.num_qubits
        qi = int(qubit_index)
        if initial_basis.upper() == "Z":
            init = PauliOperator.single_qubit_z(n, qi)
        else:
            init = PauliOperator.single_qubit_x(n, qi)
        conj = conjugate_circuit(init, qiskit_circuit)
        axis = conj.get_qubit_pauli(qi)
        phase = conj.phase_sign()
        op_str = conj.to_string()
        if phase < 0:
            op_str = f"-{op_str}"
        return {
            "axis": axis if axis in ("Z", "X") else initial_basis.upper(),
            "phase": int(phase),
            "operator_string": op_str,
            "pauli_operator": conj,
        }
