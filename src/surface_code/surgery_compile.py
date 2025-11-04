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
from .surgery_ops import MeasureRound, Merge, ParityReadout, Split, CNOTOp, TerminatePatch, ResetPatch


def _allocate_ancilla(patches: Dict[str, PatchObject], ancilla_name: str = "ancilla_0", buffer: float = 1.0) -> PatchObject:
    """Create an ancilla patch by copying structure from existing data patches.
    
    The ancilla is virtually initialized in |+⟩_L (X-basis, +1 eigenstate).
    """
    if not patches:
        raise ValueError("Cannot create ancilla without existing patches")
    
    # Use the first patch as template (all patches should have same structure)
    template = next(iter(patches.values()))
    
    # Calculate the actual bounding box of the patch
    x_coords = [x for x, y in template.coords.values()]
    y_coords = [y for x, y in template.coords.values()]
    
    patch_width = max(x_coords) - min(x_coords) if x_coords else 0
    patch_height = max(y_coords) - min(y_coords) if y_coords else 0
    
    # Add buffer spacing to avoid overlap
    # Use the larger dimension plus buffer to ensure no overlap
   
    y_offset = max(patch_width, patch_height) + float(buffer)
    
    # Create ancilla with same structure but shifted coordinates
    ancilla_coords = {q: (x, y + y_offset) for q, (x, y) in template.coords.items()}
    
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
    distance: int,
    patches: Dict[str, PatchObject],
) -> Dict[Tuple[str, str, str], List[Tuple[int, int]]]:
    """Generate default seam pairs for Control–Ancilla and Ancilla–Target.

    Important: In a rotated planar/heavy-hex patch, rough boundaries live on
    the west/east extremes (x ≈ min/max), while smooth boundaries live on the
    south/north extremes (y ≈ min/max). To realize a CNOT via lattice surgery
    you must use a rough (ZZ) merge on one boundary type and a smooth (XX)
    merge on the orthogonal boundary type. Earlier heuristics incorrectly used
    two x-extrema for both merges, which implied the ancilla had rough on one
    x-side and smooth on the other—physically inconsistent. We fix that by:
      - Rough seam: pair x-extreme columns (control vs ancilla)
      - Smooth seam: pair y-extreme rows (ancilla vs target)
    The coordinates need not be adjacent; the seam explicitly defines which
    qubits participate in the joint checks.
    """

    def _extrema_sorted(coords: Dict[int, Tuple[float, float]], axis: str, side: str):
        xs = [x for x, _ in coords.values()]
        ys = [y for _, y in coords.values()]
        tol = 0.15
        if axis == "x":
            edge = min(xs) if side == "min" else max(xs)
            sel = [(i, xy) for i, xy in coords.items() if abs(xy[0] - edge) < tol]
            # sort along the orthogonal direction (y)
            return sorted(sel, key=lambda p: p[1][1])
        else:
            edge = min(ys) if side == "min" else max(ys)
            sel = [(i, xy) for i, xy in coords.items() if abs(xy[1] - edge) < tol]
            # sort along the orthogonal direction (x)
            return sorted(sel, key=lambda p: p[1][0])

    new_seams: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}

    # Try to copy any explicit patterns first (user override wins)
    ct_rough_key = ("rough", control, target)
    ct_smooth_key = ("smooth", control, target)

    # Rough ZZ seam between control and ancilla (use x-extrema)
    if ct_rough_key in existing_seams:
        new_seams[("rough", control, ancilla)] = existing_seams[ct_rough_key].copy()
    else:
        ctrl = patches.get(control)
        anc = patches.get(ancilla)
        if ctrl and anc:
            ctrl_coords = ctrl.coords
            anc_coords = anc.coords
            # Prefer pairing the same side (e.g., both west columns) to keep
            # ordering consistent regardless of relative placement.
            ctrl_edge = _extrema_sorted(ctrl_coords, axis="x", side="min")
            anc_edge = _extrema_sorted(anc_coords, axis="x", side="min")
            pairs: List[Tuple[int, int]] = []
            for k in range(min(len(ctrl_edge), len(anc_edge), int(distance))):
                pairs.append((ctrl_edge[k][0], anc_edge[k][0]))
            new_seams[("rough", control, ancilla)] = pairs
        else:
            new_seams[("rough", control, ancilla)] = [(i, i) for i in range(int(distance))]

    # Smooth XX seam between ancilla and target (use y-extrema)
    if ct_smooth_key in existing_seams:
        new_seams[("smooth", ancilla, target)] = existing_seams[ct_smooth_key].copy()
    else:
        anc = patches.get(ancilla)
        tgt = patches.get(target)
        if anc and tgt:
            anc_coords = anc.coords
            tgt_coords = tgt.coords
            # Use north/top rows for both to respect smooth boundaries
            anc_row = _extrema_sorted(anc_coords, axis="y", side="max")
            tgt_row = _extrema_sorted(tgt_coords, axis="y", side="max")
            pairs: List[Tuple[int, int]] = []
            for k in range(min(len(anc_row), len(tgt_row), int(distance))):
                pairs.append((anc_row[k][0], tgt_row[k][0]))
            new_seams[("smooth", ancilla, target)] = pairs
        else:
            new_seams[("smooth", ancilla, target)] = [(i, i) for i in range(int(distance))]

    return new_seams


def compile_circuit_to_surgery(
    qc: QuantumCircuit,
    patches: Dict[str, PatchObject],
    seams: Dict[Tuple[str, str, str], List[Tuple[int, int]]],
    distance: int,
    bracket_map: Dict[str, str],
    warmup_rounds: int = 1,
    ancilla_strategy: str = "serialize",
    ancilla_buffer: float = 1.0,
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
    ancilla_buffer: buffer spacing between ancilla and template patch
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
    
    # Track which qubits have been terminated (measured mid-circuit)
    terminated_patches = set()
    
    # Parse gates in order; map CNOTs to ancilla-mediated surgery
    for ci in qc.data:
        name = ci.operation.name.lower()
        
        # Handle mid-circuit measurements
        if name == "measure":
            qubit_idx = qc.find_bit(ci.qubits[0]).index
            qubit_name = f"q{qubit_idx}"
            
            # Get classical register info if available
            creg_name = None
            if hasattr(ci, 'clbits') and ci.clbits:
                try:
                    creg_info = qc.find_bit(ci.clbits[0])
                    creg_name = creg_info.registers[0][0].name if creg_info.registers else None
                except Exception:
                    pass
            
            # Mark this patch as terminated
            if qubit_name not in terminated_patches:
                terminated_patches.add(qubit_name)
                ops.append(TerminatePatch(qubit_name, creg_name))
            continue
            
        if name in {"cx", "cz", "cnot"}:  # treat all as CNOT-style surgery
            control = qc.find_bit(ci.qubits[0]).index
            target = qc.find_bit(ci.qubits[1]).index
            control_name = f"q{control}"
            target_name = f"q{target}"
            
            # Create ancilla patch on first CNOT if not already created
            if not ancilla_created:
                ancilla_patch = _allocate_ancilla(patches, ancilla_name, ancilla_buffer)
                layout.add_patch(ancilla_name, ancilla_patch)
                ancilla_created = True
            
            # Build combined patches dictionary including ancilla for seam generation
            all_patches = patches.copy()
            if ancilla_name in layout.patches:
                all_patches[ancilla_name] = layout.patches[ancilla_name]
            
            # Ensure ancilla seams exist for this control/target pair.
            ancilla_seams = _auto_generate_seams(
                control_name, target_name, ancilla_name, seams, distance, all_patches
            )
            for seam_key, seam_pairs in ancilla_seams.items():
                kind, a_name, b_name = seam_key
                layout.register_seam(kind, a_name, b_name, seam_pairs)

            # Expand CNOT into ancilla-mediated surgery sequence:
            # 1. Rough ZZ merge (Control-Ancilla)
            rough_rounds = max(0, int(distance))
            ops.append(Merge("rough", control_name, ancilla_name, rounds=rough_rounds))
            for _ in range(rough_rounds):
                ops.append(MeasureRound())
            ops.append(Split("rough", control_name, ancilla_name))
            ops.append(ParityReadout("ZZ", "ZZ", control_name, ancilla_name))
            # Reset ancilla back to |+> (X eigenstate) before engaging smooth seam
            ops.append(ResetPatch(ancilla_name, basis="X"))
            # 2. Smooth XX merge (Ancilla-Target)  
            smooth_rounds = max(0, int(distance))
            ops.append(Merge("smooth", ancilla_name, target_name, rounds=smooth_rounds))
            for _ in range(smooth_rounds):
                ops.append(MeasureRound())
            ops.append(Split("smooth", ancilla_name, target_name))
            ops.append(ParityReadout("XX", "XX", ancilla_name, target_name))
            
        else:
            # Ignore 1Q gates, but still add d rounds of error correction measurements
            for _ in range(distance):
                ops.append(MeasureRound())

    return layout, ops
