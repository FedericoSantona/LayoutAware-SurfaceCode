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

from .layout import Layout, PatchObject
from .surgery_ops import MeasureRound, Merge, ParityReadout, Split, CNOTOp


def _allocate_ancilla(patches: Dict[str, PatchObject], ancilla_name: str = "ancilla_0") -> PatchObject:
    """Create an ancilla patch by copying structure from existing data patches.
    
    The ancilla is virtually initialized in |+⟩_L (X-basis, +1 eigenstate).
    """
    if not patches:
        raise ValueError("Cannot create ancilla without existing patches")
    
    # Use the first patch as template (all patches should have same structure)
    template = next(iter(patches.values()))
    
    # Create ancilla with same structure but shifted coordinates
    # Place ancilla at offset (0, 1) relative to template
    ancilla_coords = {q: (x, y + 1.0) for q, (x, y) in template.coords.items()}
    
    return PatchObject(
        n=template.n,
        z_stabs=template.z_stabs.copy(),
        x_stabs=template.x_stabs.copy(),
        logical_z=template.logical_z,
        logical_x=template.logical_x,
        coords=ancilla_coords,
        boundaries=template.boundaries,
    )


def _auto_generate_seams(
    control: str, 
    target: str, 
    ancilla: str,
    existing_seams: Dict[Tuple[str, str, str], List[Tuple[int, int]]],
    distance: int
) -> Dict[Tuple[str, str, str], List[Tuple[int, int]]]:
    """Generate default seam pairs for Control-Ancilla and Ancilla-Target merges.
    
    Default strategy: copy the Control-Target seam pattern if it exists,
    otherwise use simple zipper pattern (i,i) for i in [0..d-1].
    """
    new_seams = {}
    
    # Try to copy existing C-T pattern first
    ct_rough_key = ("rough", control, target)
    ct_smooth_key = ("smooth", control, target)
    
    # Control-Ancilla rough seams (ZZ)
    if ct_rough_key in existing_seams:
        new_seams[("rough", control, ancilla)] = existing_seams[ct_rough_key].copy()
    else:
        # Default zipper pattern
        new_seams[("rough", control, ancilla)] = [(i, i) for i in range(distance)]
    
    # Ancilla-Target smooth seams (XX)  
    if ct_smooth_key in existing_seams:
        new_seams[("smooth", ancilla, target)] = existing_seams[ct_smooth_key].copy()
    else:
        # Default zipper pattern
        new_seams[("smooth", ancilla, target)] = [(i, i) for i in range(distance)]
    
    return new_seams


def compile_circuit_to_surgery(
    qc: QuantumCircuit,
    patches: Dict[str, PatchObject],
    seams: Dict[Tuple[str, str, str], List[Tuple[int, int]]],
    distance: int,
    bracket_map: Dict[str, str],
    warmup_rounds: int = 1,
    ancilla_strategy: str = "serialize",
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
    ancilla_strategy: 'serialize' (reuse one ancilla) or 'parallelize' (multiple ancillas)
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

    # Track ancilla allocation for CNOT operations
    ancilla_created = False
    ancilla_name = "ancilla_0"
    
    # Parse gates in order; map CNOTs to ancilla-mediated surgery
    for ci in qc.data:
        name = ci.operation.name.lower()
        if name in {"cx", "cz", "cnot"}:  # treat all as CNOT-style surgery
            control = qc.find_bit(ci.qubits[0]).index
            target = qc.find_bit(ci.qubits[1]).index
            control_name = f"q{control}"
            target_name = f"q{target}"
            
            # Create ancilla patch on first CNOT if not already created
            if not ancilla_created:
                ancilla_patch = _allocate_ancilla(patches, ancilla_name)
                layout.add_patch(ancilla_name, ancilla_patch)
                
                # Generate ancilla seams (C-A and A-T)
                ancilla_seams = _auto_generate_seams(
                    control_name, target_name, ancilla_name, seams, distance
                )
                for seam_key, seam_pairs in ancilla_seams.items():
                    layout.register_seam(seam_key[0], seam_key[1], seam_key[2], seam_pairs)
                
                ancilla_created = True
            
            # Expand CNOT into ancilla-mediated surgery sequence:
            # 1. Rough ZZ merge (Control-Ancilla)
            ops.append(Merge("rough", control_name, ancilla_name, rounds=int(distance)))
            ops.append(Split("rough", control_name, ancilla_name))
            ops.append(ParityReadout("ZZ", "ZZ", control_name, ancilla_name))
            
            # 2. Smooth XX merge (Ancilla-Target)  
            ops.append(Merge("smooth", ancilla_name, target_name, rounds=int(distance)))
            ops.append(Split("smooth", ancilla_name, target_name))
            ops.append(ParityReadout("XX", "XX", ancilla_name, target_name))
            
        else:
            # 1Q gates are tracked in software; no scheduling here.
            continue

    return layout, ops


