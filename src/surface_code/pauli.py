"""Unified Pauli tracking: representation, conjugation, and frame management.

This module centralizes all Pauli-related logic:
- Pauli representation (`Pauli`)
- Conjugation through circuits (`conjugate_through_circuit`)
- Virtual-gate tracking and frame updates (`PauliTracker`)
- Small utilities (`parse_init_label`, `sequence_from_qc`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit


class Pauli:
    """Symplectic Pauli with global phase.

    Stores x_bits and z_bits bitmasks across n logical qubits and a global phase sign in {+1,-1}.
    """

    def __init__(self, n_qubits: int, x_bits: int = 0, z_bits: int = 0, phase: int = +1) -> None:
        self.n_qubits = int(n_qubits)
        self.x_bits = int(x_bits) & ((1 << self.n_qubits) - 1)
        self.z_bits = int(z_bits) & ((1 << self.n_qubits) - 1)
        self._phase = +1 if int(phase) >= 0 else -1

    @classmethod
    def single_x(cls, n_qubits: int, qubit: int) -> "Pauli":
        if qubit >= n_qubits:
            raise ValueError("Qubit out of range")
        return cls(n_qubits, x_bits=1 << int(qubit), z_bits=0, phase=+1)

    @classmethod
    def single_z(cls, n_qubits: int, qubit: int) -> "Pauli":
        if qubit >= n_qubits:
            raise ValueError("Qubit out of range")
        return cls(n_qubits, x_bits=0, z_bits=1 << int(qubit), phase=+1)

    @classmethod
    def two_xx(cls, n_qubits: int, q0: int, q1: int) -> "Pauli":
        x_bits = (1 << int(q0)) | (1 << int(q1))
        return cls(n_qubits, x_bits=x_bits, z_bits=0, phase=+1)

    @classmethod
    def two_zz(cls, n_qubits: int, q0: int, q1: int) -> "Pauli":
        z_bits = (1 << int(q0)) | (1 << int(q1))
        return cls(n_qubits, x_bits=0, z_bits=z_bits, phase=+1)

    @classmethod
    def identity(cls, n_qubits: int) -> "Pauli":
        return cls(n_qubits, 0, 0, +1)

    def copy(self) -> "Pauli":
        return Pauli(self.n_qubits, self.x_bits, self.z_bits, self._phase)

    def get_axis(self, qubit: int) -> str:
        x_bit = (self.x_bits >> int(qubit)) & 1
        z_bit = (self.z_bits >> int(qubit)) & 1
        if x_bit and z_bit:
            return "Y"
        if x_bit:
            return "X"
        if z_bit:
            return "Z"
        return "I"

    def phase_sign(self) -> int:
        return +1 if self._phase >= 0 else -1

    # Conjugations
    def conjugate_h(self, qubit: int) -> None:
        q = int(qubit)
        x_bit = (self.x_bits >> q) & 1
        z_bit = (self.z_bits >> q) & 1
        self.x_bits &= ~(1 << q)
        self.z_bits &= ~(1 << q)
        self.x_bits |= z_bit << q
        self.z_bits |= x_bit << q

    def conjugate_x(self, qubit: int) -> None:
        q = int(qubit)
        z_bit = (self.z_bits >> q) & 1
        if z_bit:
            self._phase *= -1

    def conjugate_z(self, qubit: int) -> None:
        q = int(qubit)
        x_bit = (self.x_bits >> q) & 1
        if x_bit:
            self._phase *= -1

    def conjugate_cnot(self, control: int, target: int) -> None:
        c = int(control)
        t = int(target)
        cx = (self.x_bits >> c) & 1
        self.x_bits ^= cx << t
        tz = (self.z_bits >> t) & 1
        self.z_bits ^= tz << c

    def to_string(self) -> str:
        parts: List[str] = []
        for i in range(self.n_qubits):
            axis = self.get_axis(i)
            if axis != "I":
                parts.append(f"{axis}(q{i})")
        return "*".join(parts) if parts else "I"


def conjugate_through_circuit(initial: Pauli, qc: QuantumCircuit) -> Pauli:
    """Compute U† initial U by walking qc backwards."""
    p = initial.copy()
    for inst in reversed(qc.data):
        name = inst.operation.name.lower()
        qubits = [qc.find_bit(qb).index for qb in inst.qubits]
        if name == "h":
            if len(qubits) != 1:
                raise ValueError("H must act on 1 qubit")
            p.conjugate_h(qubits[0])
        elif name in {"cx", "cnot", "cz"}:
            if len(qubits) != 2:
                raise ValueError("CNOT must act on 2 qubits")
            p.conjugate_cnot(qubits[0], qubits[1])
        elif name == "x":
            if len(qubits) != 1:
                raise ValueError("X must act on 1 qubit")
            p.conjugate_x(qubits[0])
        elif name == "z":
            if len(qubits) != 1:
                raise ValueError("Z must act on 1 qubit")
            p.conjugate_z(qubits[0])
        else:
            # Ignore barriers, measures, etc.
            continue
    return p


class PauliTracker:
    """Track virtual gates and Pauli frame bits across logical qubits."""

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = int(n_qubits)
        self.frame: Dict[str, Dict[str, Any]] = {f"q{i}": {"fx": 0, "fz": 0} for i in range(self.n_qubits)}
        self.virtual_gates: Dict[str, List[str]] = {f"q{i}": [] for i in range(self.n_qubits)}

    # Virtual gates
    def add_virtual_gate(self, qubit_index: int, gate: str) -> None:
        qn = f"q{int(qubit_index)}"
        self.virtual_gates[qn].append(gate.upper())
        self._fold_sequence_into_frame(qubit_index)

    def set_sequence(self, qubit_index: int, gates: Iterable[str]) -> None:
        qn = f"q{int(qubit_index)}"
        self.virtual_gates[qn] = [str(g).upper() for g in gates]
        self._fold_sequence_into_frame(qubit_index)

    def get_sequence(self, qubit_index: int) -> List[str]:
        return list(self.virtual_gates[f"q{int(qubit_index)}"])

    @staticmethod
    def conjugate_axis_by_sequence(axis: str, gates: List[str]) -> Tuple[str, int]:
        axis = axis.upper()
        p = Pauli.single_z(1, 0) if axis == "Z" else Pauli.single_x(1, 0)
        for g in reversed(gates or []):
            ug = str(g).upper()
            if ug == "H":
                p.conjugate_h(0)
            elif ug == "X":
                p.conjugate_x(0)
            elif ug == "Z":
                p.conjugate_z(0)
        return p.get_axis(0), p.phase_sign()

    def _fold_sequence_into_frame(self, qubit_index: int) -> None:
        qn = f"q{int(qubit_index)}"
        seq = self.virtual_gates[qn]
        _, z_phase = self.conjugate_axis_by_sequence("Z", seq)
        _, x_phase = self.conjugate_axis_by_sequence("X", seq)
        self.frame[qn]["fx"] = 1 if z_phase < 0 else 0
        self.frame[qn]["fz"] = 1 if x_phase < 0 else 0

    # Frame updates
    def update_cnot(self, control: str, target: str, m_zz: np.ndarray, m_xx: np.ndarray) -> None:
        self.frame[target]["fz"] ^= m_zz
        self.frame[control]["fx"] ^= m_xx

    def frame_bit(self, qubit_name: str, axis: str) -> int:
        key = "fx" if axis.upper() == "Z" else "fz"
        v = self.frame.get(qubit_name, {}).get(key, 0)
        if isinstance(v, np.ndarray):
            return int(round(float(v.mean()))) & 1
        return int(v) & 1

    # Final operator info
    def final_operator_info(self, qubit_index: int, initial_basis: str, qc: QuantumCircuit) -> Dict[str, Any]:
        n = int(qc.num_qubits)
        qi = int(qubit_index)
        init = Pauli.single_z(n, qi) if initial_basis.upper() == "Z" else Pauli.single_x(n, qi)
        conj = conjugate_through_circuit(init, qc)
        axis = conj.get_axis(qi)
        phase = conj.phase_sign()
        op_str = conj.to_string()
        if phase < 0:
            op_str = f"-{op_str}"
        return {
            "axis": axis if axis in ("Z", "X") else initial_basis.upper(),
            "phase": int(phase),
            "operator_string": op_str,
            "pauli": conj,
        }

    # Expected flips (phase-based)
    @staticmethod
    def expected_flip(initial_basis: str, gates: Iterable[str]) -> int:
        _, phase = PauliTracker.conjugate_axis_by_sequence(initial_basis, list(gates))
        return 1 if phase < 0 else 0

    # Bit corrections (reporting policy)
    def apply_corrections(
        self,
        bits: np.ndarray,
        qubit_name: str,
        operator_axis: str,
        operator_phase: int,
        decoder_flips: Optional[np.ndarray] = None,
        snapshot_basis: Optional[str] = None,
    ) -> np.ndarray:
        out = np.array(bits, dtype=np.uint8, copy=True)
        # Frame flip keyed by actual operator axis
        axis = operator_axis.upper()
        frame_key = "fx" if axis == "Z" else "fz"
        flip = self.frame.get(qubit_name, {}).get(frame_key, 0)
        if isinstance(flip, np.ndarray):
            out ^= flip.astype(np.uint8)
        elif int(flip) & 1:
            out ^= np.ones_like(out, dtype=np.uint8)
        # Phase flip (avoid double count if desired by caller)
        if int(operator_phase) < 0:
            out ^= np.ones_like(out, dtype=np.uint8)
        # Decoder flips only if operator axis matches printed snapshot basis
        if decoder_flips is not None and snapshot_basis is not None and axis == snapshot_basis.upper():
            arr = decoder_flips.astype(np.uint8)
            out ^= arr
        return out


# Utilities
def parse_init_label(label: str) -> Tuple[str, int]:
    s = (label or "0").strip()
    if s == "0":
        return "Z", +1
    if s == "1":
        return "Z", -1
    if s == "+":
        return "X", +1
    if s == "-":
        return "X", -1
    raise ValueError("init label must be one of '0','1','+','-'")


def sequence_from_qc(qc: QuantumCircuit, allowed: Optional[Iterable[str]] = None, per_qubit: bool = True) -> Dict[str, List[str]]:
    """Extract per-qubit sequences of logical single-qubit gates from a Qiskit circuit.

    allowed: set of lowercase names (default: {x,z,h})
    """
    allowed_set = set(a.lower() for a in (allowed or {"x", "z", "h"}))
    seqs: Dict[str, List[str]] = {f"q{i}": [] for i in range(qc.num_qubits)}
    for inst in qc.data:
        name = inst.operation.name.lower()
        if name not in allowed_set:
            continue
        for qb in inst.qubits:
            qidx = qc.find_bit(qb).index
            seqs[f"q{qidx}"].append(name.upper())
    return seqs


