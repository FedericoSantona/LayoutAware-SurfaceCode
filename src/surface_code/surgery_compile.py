"""Compile a logical circuit into a surgery layout and timeline.

Assumptions (v1):
  - One patch per logical qubit (caller supplies `PatchObject`s and seams).
  - CNOT(control,target) ⇒ rough ZZ merge for d rounds, split, then smooth XX
    merge for d rounds, split, with ParityReadout markers for ZZ and XX.
  - Single-qubit gates (X, Z, H) remain virtual and are tracked in Pauli frame
    elsewhere; this compiler only schedules stabilizer measurement structure.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from qiskit import QuantumCircuit

from .multi_patch import Layout, PatchObject
from .surgery_ops import MeasureRound, Merge, ParityReadout, Split


def compile_circuit_to_surgery(
    qc: QuantumCircuit,
    patches: Dict[str, PatchObject],
    seams: Dict[Tuple[str, str, str], List[Tuple[int, int]]],
    distance: int,
    bracket_map: Dict[str, str],
    warmup_rounds: int = 1,
) -> Tuple[Layout, List[object]]:
    """Return a `Layout` and ops timeline for surgery execution.

    Parameters
    ----------
    qc: QuantumCircuit with 1Q/2Q gates
    patches: dict mapping logical labels (e.g., 'q0', 'q1') to PatchObject
    seams: dict keyed by (kind, a, b) with lists of local seam pairs
    distance: number of rounds per merge window
    bracket_map: per-patch bracket basis ('Z'|'X') used by the DEM
    warmup_rounds: initial `MeasureRound` cycles to establish references
    """
    layout = Layout()
    # Insert patches in wire order
    qubit_labels: List[str] = [f"q{i}" for i in range(qc.num_qubits)]
    for label in qubit_labels:
        if label not in patches:
            raise KeyError(f"Missing PatchObject for {label}")
        layout.add_patch(label, patches[label])
    # Register seams as provided
    for key, pairs in seams.items():
        kind, a, b = key
        layout.register_seam(kind, a, b, pairs)

    # Build timeline: warmup rounds
    ops: List[object] = []
    for _ in range(max(0, int(warmup_rounds))):
        ops.append(MeasureRound())

    # Parse gates in order; map only CNOTs to merges
    for ci in qc.data:
        name = ci.operation.name.lower()
        if name in {"cx", "cz", "cnot"}:  # treat all as CNOT-style surgery
            control = qc.find_bit(ci.qubits[0]).index
            target = qc.find_bit(ci.qubits[1]).index
            a = f"q{control}"
            b = f"q{target}"
            # Rough ZZ between control-target
            ops.append(Merge("rough", a, b, rounds=int(distance)))
            ops.append(Split("rough", a, b))
            ops.append(ParityReadout("ZZ", "ZZ", a, b))
            # Smooth XX between control-target
            ops.append(Merge("smooth", a, b, rounds=int(distance)))
            ops.append(Split("smooth", a, b))
            ops.append(ParityReadout("XX", "XX", a, b))
        else:
            # 1Q gates are tracked in software; no scheduling here.
            continue

    return layout, ops


