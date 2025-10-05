from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.transpiler import Target, PassManager, CouplingMap, InstructionDurations
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as SEL

# Passes (2.x names)
from qiskit.transpiler.passes import (
    BasisTranslator,
    Optimize1qGates,                 # merge single-qubit chains
    CommutativeCancellation,         # remove commuting gate pairs
    GateDirection,                   # enforce directed CX
)
from qiskit.transpiler.passes.layout import SabreLayout
from qiskit.transpiler.passes.routing import SabreSwap
from qiskit.transpiler.passes.scheduling import (
    ALAPScheduleAnalysis,
    ASAPScheduleAnalysis,
    PadDynamicalDecoupling,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _prune_idle_qubits_simple(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Return a copy of `qc` with qubit wires removed if they are never acted on
    by any instruction (no gates, no measure/reset). This keeps only the set
    of *active* physical qubits and remaps instructions accordingly.

    Notes:
      - This is intentionally minimal and only used for metrics.
      - QuantumRegister structure is not preserved; classical bits are kept.
    """
    # Collect active physical indices in first-seen order for stable remapping.
    active_order: List[int] = []
    active_set: set[int] = set()
    for instr in qc.data:
        for q in instr.qubits:
            idx = qc.find_bit(q).index
            if idx not in active_set:
                active_set.add(idx)
                active_order.append(idx)

    if len(active_set) == qc.num_qubits:
        return qc  # nothing to prune

    # Build a thin circuit with only active wires.
    new_qc = QuantumCircuit(len(active_order), qc.num_clbits, name=qc.name)
    if getattr(qc, "metadata", None) is not None:
        try:
            new_qc.metadata = dict(qc.metadata)
        except Exception:
            pass

    # Helper: map old qargs -> new_qargs using the active-order lookup.
    index_map = {old: new for new, old in enumerate(active_order)}
    def _map_qargs(qargs):
        return [new_qc.qubits[index_map[qc.find_bit(q).index]] for q in qargs]

    for instr in qc.data:
        new_qc.append(instr.operation, _map_qargs(instr.qubits), instr.clbits)

    return new_qc


def _allowed_cx_directions(target: Target) -> set[tuple[int, int]]:
    """Collect directed CX edges allowed by the Target."""
    allowed: set[tuple[int, int]] = set()
    num_qubits = getattr(target, "num_qubits", 0)
    for q0 in range(num_qubits):
        for q1 in range(num_qubits):
            if q0 == q1:
                continue
            try:
                supported = target.instruction_supported("cx", (q0, q1))
            except Exception:
                supported = False
            if supported:
                allowed.add((q0, q1))
    return allowed


def _count_2q_and_swaps(qc: QuantumCircuit) -> tuple[int, int]:
    """Count mapped 2Q gates and SWAPs (SWAPs included in 2Q total)."""
    ops = dict(qc.count_ops())
    twoq_names = {
        "cx", "cz", "cp", "ecr", "iswap",
        "rxx", "ryy", "rzz", "xx_plus_yy",
        "swap",
    }
    twoq = sum(int(ops.get(k, 0)) for k in twoq_names)
    swaps = int(ops.get("swap", 0))
    return twoq, swaps


def _estimate_dir_fix_fraction(qc: QuantumCircuit, target: Target) -> float:
    """
    Heuristic: fraction of CXs whose (control, target) is NOT supported
    by the Target's direction set. Assumes physical indices after layout/routing.
    """
    allowed = _allowed_cx_directions(target)
    if not allowed:
        return 0.0
    total = 0
    bad = 0
    for instr in qc.data:
        op = instr.operation
        if op.name != "cx":
            continue
        total += 1
        ctrl = qc.find_bit(instr.qubits[0]).index
        targ = qc.find_bit(instr.qubits[1]).index
        if (ctrl, targ) not in allowed:
            bad += 1
    return (bad / total) if total else 0.0


def _make_dd_sequence(policy: str) -> List[Any]:
    """Build a simple DD sequence per PadDynamicalDecoupling requirements."""
    from qiskit.circuit.library import IGate, XGate, YGate

    if policy.upper() in {"XIX", "XI X", "X-I-X"}:
        return [XGate(), IGate(), XGate(), IGate()]
    if policy.upper() in {"XYXY", "X-Y-X-Y"}:
        return [XGate(), YGate(), XGate(), YGate()]
    raise ValueError(f"Unknown dd_policy '{policy}' (use 'XIX' or 'XYXY').")


# -----------------------------------------------------------------------------
# Pure transpile steps (stateless, testable)
# -----------------------------------------------------------------------------

def unroll(qc: QuantumCircuit, target: Target, basis: Optional[Iterable[str]] = None) -> QuantumCircuit:
    """
    Rewrite to the Target's basis using the 2.x BasisTranslator and SEL.
    """
    pm = PassManager()
    basis_tuple: Tuple[str, ...]
    if basis is not None:
        basis_tuple = tuple(basis)
    else:
        # Attempt to pull basis from target; fall back to circuit ops if unavailable.
        try:
            operation_names = list(getattr(target, "operations", []))
            basis_tuple = tuple(str(op.name) for op in operation_names)
        except Exception:
            basis_tuple = tuple(sorted(set(op.name for op in qc.data)))

    try:
        pm.append(BasisTranslator(SEL, target=target, target_basis=basis_tuple))
    except TypeError:
        # Older Qiskit: BasisTranslator expects target_basis only.
        pm.append(BasisTranslator(SEL, target_basis=basis_tuple))
    return pm.run(qc)


def _coupling_map_from_target(target: Target) -> CouplingMap:
    edges = set()
    for q0, q1 in _allowed_cx_directions(target):
        edges.add((q0, q1))
        edges.add((q1, q0))
    if not edges:
        # Fallback to a line topology if none reported.
        edges = {(i, i + 1) for i in range(max(0, getattr(target, "num_qubits", 0) - 1))}
        edges |= {(b, a) for (a, b) in edges}
    return CouplingMap(list(edges))


def initial_layout(qc: QuantumCircuit, target: Target, seed: int, max_iterations: int = 5) -> QuantumCircuit:
    """Choose initial placement with SabreLayout (best for sparse heavy-hex)."""
    pm = PassManager()
    try:
        layout_pass = SabreLayout(target=target, seed=seed, max_iterations=max_iterations)
    except TypeError:
        coupling_map = _coupling_map_from_target(target)
        layout_pass = SabreLayout(coupling_map=coupling_map, seed=seed, max_iterations=max_iterations)
    pm.append(layout_pass)
    return pm.run(qc)


def route(qc: QuantumCircuit, target: Target, seed: int) -> QuantumCircuit:
    """
    Insert SWAPs to satisfy connectivity using SabreSwap.
    """
    pm = PassManager()
    try:
        swap_pass = SabreSwap(target=target, seed=seed)
    except TypeError:
        coupling_map = _coupling_map_from_target(target)
        swap_pass = SabreSwap(coupling_map=coupling_map, seed=seed)
    pm.append(swap_pass)
    return pm.run(qc)


def gate_direction(qc: QuantumCircuit, target: Target) -> QuantumCircuit:
    """
    Enforce directed CX orientation according to Target.
    """
    pm = PassManager()
    try:
        pm.append(GateDirection(target=target))
    except TypeError:
        coupling_map = _coupling_map_from_target(target)
        pm.append(GateDirection(coupling_map=coupling_map))
    return pm.run(qc)


def opt_local(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Lightweight local cleanups that preserve routing.
    """
    pm = PassManager()
    pm.append(Optimize1qGates())
    pm.append(CommutativeCancellation())
    return pm.run(qc)


def _augmented_durations(qc: QuantumCircuit, target: Target) -> InstructionDurations:
    """Return a copy of the target durations with fallbacks for non-native ops.

    Heavy-hex fake targets typically omit timing data for composite instructions
    like SWAP and for control-flow ops such as IF/ELSE. We synthesize reasonable
    values (3 * CX for SWAP; 0 for control-flow) so the scheduler can proceed.
    """

    durations = InstructionDurations()
    durations.update(target.durations())

    units = durations.units_used()
    if "dt" in units and len(units) == 1:
        preferred_unit = "dt"
    elif "s" in units:
        preferred_unit = "s"
    elif "dt" in units:
        preferred_unit = "dt"
    else:
        preferred_unit = "dt" if getattr(target, "dt", None) is not None else "s"

    extras: List[Tuple[str, List[int] | None, float, str]] = []

    for instr in qc.data:
        op = instr.operation
        name = op.name
        qubit_indices = [qc.find_bit(q).index for q in instr.qubits]

        # Skip if the target already specifies a duration for this op/qubits.
        try:
            durations.get(name, qubit_indices or [0], unit=preferred_unit)
            continue
        except TranspilerError:
            pass

        if name == "swap" and len(qubit_indices) == 2:
            try:
                cx_dur = durations.get("cx", qubit_indices, unit=preferred_unit)
            except TranspilerError:
                continue
            extras.append((name, qubit_indices, 3 * cx_dur, preferred_unit))
        elif name in {"if_else", "while_loop", "for_loop", "switch_case"}:
            extras.append((name, qubit_indices or None, 0.0, preferred_unit))

    if extras:
        durations.update(extras)

    return durations


def schedule(qc: QuantumCircuit, target: Target, mode: str = "alap", dd_policy: Optional[str] = None) -> QuantumCircuit:
    """
    Hardware-aware scheduling. Adds timing metadata; optional DD on idle gaps.
    NOTE: caller must ensure barriers around QEC A/B/C/D half-rounds if dd_between_rounds_only is desired.
    """
    pm = PassManager()
    durations = _augmented_durations(qc, target)

    if mode.lower() == "alap":
        try:
            pm.append(ALAPScheduleAnalysis(durations=durations))
        except TypeError:
            # Older Qiskit releases exposed the kwarg as instruction_durations.
            pm.append(ALAPScheduleAnalysis(instruction_durations=durations))
    elif mode.lower() == "asap":
        try:
            pm.append(ASAPScheduleAnalysis(durations=durations))
        except TypeError:
            pm.append(ASAPScheduleAnalysis(instruction_durations=durations))
    else:
        raise ValueError("schedule(mode=...) must be 'alap' or 'asap'.")

    if dd_policy:
        seq = _make_dd_sequence(dd_policy)
        pm.append(PadDynamicalDecoupling(durations=durations, dd_sequence=seq))

    return pm.run(qc)


def score(qc: QuantumCircuit, target: Target) -> Dict[str, Any]:
    """
    Compute core metrics with consistent counting rules.
    - depth: includes measurement/reset; respects barriers.
    - duration_ns: set if scheduling populated circuit.duration (dt-aware backends). Otherwise None.
    - twoq, swaps: mapped counts.
    - dir_fixes: heuristic fraction of CX that violate target direction (pre-fix circuits).
    - n_qubits: RESERVED physical qubits (post-layout/routing wires, even if idle).
    - n_qubits_active: ACTIVE physical qubits (after pruning idle wires).
    """
    twoq, swaps = _count_2q_and_swaps(qc)

    # Reserved = as-routed wire count
    n_qubits_reserved = qc.num_qubits

    # Active = after pruning wires that never appear in any instruction
    try:
        qc_pruned = _prune_idle_qubits_simple(qc)
        n_qubits_active = qc_pruned.num_qubits
    except Exception:
        n_qubits_active = n_qubits_reserved

    metrics: Dict[str, Any] = {
        "n_qubits_reserved": n_qubits_reserved,
        "n_qubits_active": n_qubits_active,
        "depth": int(qc.depth()),
        "twoq": twoq,
        "swaps": swaps,
        "dir_fixes": _estimate_dir_fix_fraction(qc, target),
        "duration_ns": None,
    }
    # Qiskit sets duration in dt; convert to ns if circuit has .duration and target has dt
    try:
        if getattr(qc, "duration", None) is not None and hasattr(target, "dt") and target.dt is not None:
            metrics["duration_ns"] = float(qc.duration * target.dt * 1e9)
    except Exception:
        pass
    return metrics
