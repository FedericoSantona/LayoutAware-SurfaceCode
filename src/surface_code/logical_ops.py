"""Logical-frame helpers for single-qubit logical gate sequences.

Implements a minimal byproduct-tracking frame for logical X/Z/H gates and
utilities for mapping a 1-qubit Qiskit circuit (only H, X, Z) into an end-basis
and expected observable flip parity for correlation measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from qiskit.circuit import QuantumCircuit


@dataclass
class LogicalFrame:
    fx: int = 0         # pending X-byproduct parity (flips X_L readout)
    fz: int = 0         # pending Z-byproduct parity (flips Z_L readout)
    swap_xz: bool = False  # orientation bit (odd number of H so far)


def logical_X(frame: LogicalFrame) -> None:
    frame.fz ^= 1   # X anticommutes with Z_L


def logical_Z(frame: LogicalFrame) -> None:
    frame.fx ^= 1   # Z anticommutes with X_L


def logical_H(frame: LogicalFrame) -> None:
    frame.fx, frame.fz = frame.fz, frame.fx   # swap byproducts
    frame.swap_xz ^= 1                         # flip orientation going forward


def apply_sequence(frame: LogicalFrame, seq: Iterable[str]) -> LogicalFrame:
    """Apply a sequence like ["H","X","Z"] to the frame in-place and return it.

    Gate symbols are case-insensitive and must be a subset of {"X","Z","H"}.
    """
    for g in seq:
        s = g.upper()
        if s == "X":
            logical_X(frame)
        elif s == "Z":
            logical_Z(frame)
        elif s == "H":
            logical_H(frame)
        else:
            raise ValueError(f"Unsupported logical gate '{g}'. Only X, Z, H are allowed.")
    return frame


def end_basis_and_flip(initial_basis: str, frame: LogicalFrame) -> Tuple[str, int]:
    """Return (end_basis, expected_parity_flip) after applying frame updates.

    initial_basis: 'Z' or 'X' designating which logical observable is tracked at t0.
    The end basis swaps when an odd number of H is applied. The expected flip is
    the appropriate byproduct parity for the basis measured at the end.
    """
    b0 = initial_basis.upper().strip()
    if b0 not in {"Z", "X"}:
        raise ValueError("initial_basis must be 'Z' or 'X'")
    # Swap orientation on odd H counts.
    end_is_Z = (b0 == "Z") ^ bool(frame.swap_xz)
    end_basis = "Z" if end_is_Z else "X"
    flip = frame.fz if end_is_Z else frame.fx
    return end_basis, int(flip & 1)


def circuit_to_gates(qiskit_circuit) -> List[str]:
    """Extract a compact gate list ['H','X','Z', ...] from a 1-qubit Qiskit circuit.

    Raises if more than one qubit is used or unsupported instructions are present.
    Measurements, barriers, and idles are ignored.
    """

    if not isinstance(qiskit_circuit, QuantumCircuit):
        raise TypeError("Expected a qiskit.circuit.QuantumCircuit")
    if qiskit_circuit.num_qubits != 1:
        raise ValueError("The simple logical benchmark must be a single-qubit circuit.")

    allowed = {"x": "X", "z": "Z", "h": "H"}
    ignore = {"barrier", "measure", "reset", "delay", "id"}

    seq: List[str] = []
    for ci in qiskit_circuit.data:
        name = ci.operation.name
        if name in ignore:
            continue
        if name not in allowed:
            raise ValueError(f"Unsupported op '{name}' in simple 1Q circuit. Only H/X/Z allowed.")
        # Assert all ops target qubit 0
        if len(ci.qubits) != 1 or qiskit_circuit.find_bit(ci.qubits[0]).index != 0:
            raise ValueError("All ops must act on the single logical qubit (index 0).")
        seq.append(allowed[name])
    return seq


def parse_init_label(label: str) -> Tuple[str, int]:
    """Parse an init label ("0","1","+","-") into (basis, sign).

    Returns:
        (basis, sign) where basis is "Z" or "X" and sign is +1 ("+" eigenstate) or -1 ("-" eigenstate).
    """
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


@dataclass
class PauliFrame:
    """Track virtual Pauli corrections (the Pauli frame) for a single logical qubit.

    A Pauli frame records pending logical flips that should be interpreted
    virtually instead of physically applying X/Z corrections. Each tracked
    basis stores a parity bit (0 -> no flip, 1 -> flip).
    """

    flips: Dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _normalize_basis(basis: str) -> str:
        b = basis.upper()
        if b not in {"X", "Z"}:
            raise ValueError("basis must be 'X' or 'Z'")
        return b

    def set_flip(self, basis: str, value: int) -> None:
        """Set the tracked flip bit for a logical basis."""
        b = self._normalize_basis(basis)
        self.flips[b] = int(value) & 1

    def toggle_flip(self, basis: str, value: int = 1) -> None:
        """XOR the tracked flip bit for a basis with ``value`` (default 1)."""
        b = self._normalize_basis(basis)
        current = self.flips.get(b, 0)
        self.flips[b] = (current ^ (int(value) & 1)) & 1

    def get_flip(self, basis: str) -> int:
        """Return the pending flip bit for ``basis`` (0 when unset)."""
        b = self._normalize_basis(basis)
        return int(self.flips.get(b, 0)) & 1

    def interpret_measurement(self, measurement_bit: int, basis: str) -> int:
        """Apply the tracked flip for ``basis`` to a logical measurement bit."""
        return (int(measurement_bit) ^ self.get_flip(basis)) & 1

    def copy(self) -> "PauliFrame":
        """Return a shallow copy of this Pauli frame."""
        return PauliFrame(flips=dict(self.flips))
