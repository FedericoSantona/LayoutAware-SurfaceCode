"""Pauli conjugation engine for frame-aware demo measurements.

This module implements a symplectic Pauli tracker that computes U†σU for demo operators,
where U is the logical circuit. This makes demos reflect the "as-if physical" state
after virtual gates are applied in the Pauli frame.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
import stim
from qiskit.circuit import QuantumCircuit


class PauliOperator:
    """Represents a Pauli operator as symplectic tableau (x, z) bitsets.
    
    For n logical qubits, represents a Pauli product by two length-n bitsets:
    - x[i] = 1 means X acts on logical qubit i
    - z[i] = 1 means Z acts on logical qubit i
    """
    
    def __init__(self, n_qubits: int, x_bits: int = 0, z_bits: int = 0):
        """Initialize Pauli operator with n qubits.
        
        Args:
            n_qubits: Number of logical qubits
            x_bits: Bitmask for X operators (bit i = 1 if X acts on qubit i)
            z_bits: Bitmask for Z operators (bit i = 1 if Z acts on qubit i)
        """
        self.n_qubits = n_qubits
        self.x_bits = x_bits & ((1 << n_qubits) - 1)  # Mask to n_qubits bits
        self.z_bits = z_bits & ((1 << n_qubits) - 1)  # Mask to n_qubits bits
    
    @classmethod
    def single_qubit_x(cls, n_qubits: int, qubit: int) -> 'PauliOperator':
        """Create single-qubit X operator on specified qubit."""
        if qubit >= n_qubits:
            raise ValueError(f"Qubit {qubit} out of range for {n_qubits} qubits")
        return cls(n_qubits, x_bits=1 << qubit, z_bits=0)
    
    @classmethod
    def single_qubit_z(cls, n_qubits: int, qubit: int) -> 'PauliOperator':
        """Create single-qubit Z operator on specified qubit."""
        if qubit >= n_qubits:
            raise ValueError(f"Qubit {qubit} out of range for {n_qubits} qubits")
        return cls(n_qubits, x_bits=0, z_bits=1 << qubit)
    
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
        """Apply X conjugation: no-op for operator tracking (phase only)."""
        pass  # X gates only add byproduct phases, don't change operator structure
    
    def conjugate_Z(self, qubit: int) -> None:
        """Apply Z conjugation: no-op for operator tracking (phase only)."""
        pass  # Z gates only add byproduct phases, don't change operator structure
    
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
        return PauliOperator(self.n_qubits, self.x_bits, self.z_bits)


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


def map_logical_to_physical_string(
    patch: Any,  # Patch object
    logical_pauli: str, 
    swap_xz: bool
) -> str:
    """Map logical operator to physical string using frame state.
    
    Args:
        patch: Patch object containing logical_x and logical_z strings
        logical_pauli: 'Z' or 'X' - the logical operator to map
        swap_xz: True if H has been applied (odd number of times)
        
    Returns:
        The appropriate physical string from patch.logical_x or patch.logical_z
    """
    if logical_pauli == "Z":
        if swap_xz:
            return patch.logical_x  # Z maps to X_L after H
        else:
            return patch.logical_z  # Z maps to Z_L normally
    elif logical_pauli == "X":
        if swap_xz:
            return patch.logical_z  # X maps to Z_L after H
        else:
            return patch.logical_x  # X maps to X_L normally
    else:
        raise ValueError(f"logical_pauli must be 'Z' or 'X', got '{logical_pauli}'")


def pauli_to_physical_mpp(
    pauli_op: PauliOperator, 
    layout: Any,  # Layout object
    bracket_map: Dict[str, str]
) -> List[Tuple[int, str]]:
    """Convert logical Pauli operator to physical qubit positions for Stim MPP.
    
    Args:
        pauli_op: The conjugated Pauli operator
        layout: Layout object containing patch information
        bracket_map: Mapping from patch names to basis ('Z' or 'X')
        
    Returns:
        List of (global_qubit_idx, pauli_char) tuples for Stim MPP
    """
    physical_targets = []
    
    # Map logical qubits to patches
    patch_names = sorted(bracket_map.keys())
    if len(patch_names) != pauli_op.n_qubits:
        raise ValueError(f"Mismatch: {len(patch_names)} patches vs {pauli_op.n_qubits} logical qubits")
    
    # Get offsets for each patch
    offsets = layout.offsets()
    
    for logical_qubit in range(pauli_op.n_qubits):
        patch_name = patch_names[logical_qubit]
        pauli_char = pauli_op.get_qubit_pauli(logical_qubit)
        
        if pauli_char == "I":
            continue  # Skip identity
        
        # Get the logical operator string for this patch
        patch = layout.patches[patch_name]
        if pauli_char == "X":
            logical_string = patch.logical_x
        elif pauli_char == "Z":
            logical_string = patch.logical_z
        elif pauli_char == "Y":
            # For Y, we need both X and Z components
            # This will be handled by resolving collisions below
            logical_string_x = patch.logical_x
            logical_string_z = patch.logical_z
        else:
            continue
        
        # Map logical string positions to global physical qubits
        base_offset = offsets[patch_name]
        
        if pauli_char == "Y":
            # Handle Y by combining X and Z components
            for i, (x_char, z_char) in enumerate(zip(logical_string_x, logical_string_z)):
                if x_char == "X" and z_char == "Z":
                    physical_targets.append((base_offset + i, "Y"))
                elif x_char == "X":
                    physical_targets.append((base_offset + i, "X"))
                elif z_char == "Z":
                    physical_targets.append((base_offset + i, "Z"))
        else:
            # Handle X or Z
            for i, char in enumerate(logical_string):
                if char == pauli_char:
                    physical_targets.append((base_offset + i, pauli_char))
    
    # Resolve collisions: combine operators on same physical qubit
    qubit_ops = {}  # global_qubit_idx -> set of pauli_chars
    
    for global_idx, pauli_char in physical_targets:
        if global_idx not in qubit_ops:
            qubit_ops[global_idx] = set()
        qubit_ops[global_idx].add(pauli_char)
    
    # Resolve collisions per qubit
    final_targets = []
    for global_idx, pauli_chars in qubit_ops.items():
        if len(pauli_chars) == 1:
            final_targets.append((global_idx, list(pauli_chars)[0]))
        elif pauli_chars == {"X", "Z"}:
            final_targets.append((global_idx, "Y"))
        elif pauli_chars == {"X"}:
            final_targets.append((global_idx, "X"))
        elif pauli_chars == {"Z"}:
            final_targets.append((global_idx, "Z"))
        # If same Pauli appears twice, it cancels to identity (drop it)
    
    return final_targets
