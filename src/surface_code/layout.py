"""Multi-patch data model: relocatable patches, explicit seams, and layout.

This module defines:
  - PatchObject: a single rotated-planar/heavy-hex patch with local stabilizers
    and logicals, plus local coordinates.
  - Layout: a registry of named patches, their global placement (via offsets),
    and explicit seam definitions for surgery (pairs of touching local qubits).

Coordinates and seam definitions are supplied by the caller in v1 to avoid
inferring geometry from builders. Offsets are applied when adding patches to the
layout; global qubit indices are assigned contiguously in insertion order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Tuple


Coord = Tuple[float, float]


@dataclass(frozen=True)
class PatchObject:
    """Container for one patch (local frame) with optional boundary labels.

    Fields expect local indexing in the range [0, n).
    """

    n: int
    z_stabs: List[str]
    x_stabs: List[str]
    logical_z: str
    logical_x: str
    coords: Dict[int, Coord]
    # Optional boundary labeling; e.g., {"rough": {local_q...}, "smooth": {...}}
    boundaries: Optional[Dict[str, set[int]]] = None

    def with_offset(self, q_offset: int, x_offset: float, y_offset: float) -> "PatchObject":
        """Return a relocated copy with coordinates shifted by (x_offset, y_offset).

        The Pauli strings remain local; globalization is handled by Layout. We keep
        this method to support pre-shifting local coords before adding into a layout.
        """
        shifted_coords: Dict[int, Coord] = {
            q: (xy[0] + x_offset, xy[1] + y_offset) for q, xy in self.coords.items()
        }
        return replace(self, coords=shifted_coords)


class Layout:
    """Global placement of multiple patches and explicit seams for surgery.

    Patches are stored in insertion order; global indices are assigned by
    contiguous concatenation of each patch's local [0..n) domain.
    """

    def __init__(self) -> None:
        self._order: List[str] = []
        self.patches: Dict[str, PatchObject] = {}
        # Seam registry keyed by (kind, a, b) with kind in {"rough","smooth"}
        # Each value is a list of local-index pairs (i_in_a, j_in_b)
        self.seams: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}

    # ----- patch placement -------------------------------------------------

    def add_patch(self, name: str, patch: PatchObject) -> None:
        if name in self.patches:
            raise ValueError(f"Patch '{name}' already exists in layout")
        self._order.append(name)
        self.patches[name] = patch

    def offsets(self) -> Dict[str, int]:
        """Return the starting global qubit index for each patch, in order."""
        offsets: Dict[str, int] = {}
        cur = 0
        for name in self._order:
            offsets[name] = cur
            cur += self.patches[name].n
        return offsets

    def global_n(self) -> int:
        return sum(self.patches[name].n for name in self._order)

    def global_coords(self) -> Dict[int, Coord]:
        """Concatenate per-patch coords into a global index→coord map."""
        coords: Dict[int, Coord] = {}
        offs = self.offsets()
        for name in self._order:
            off = offs[name]
            patch = self.patches[name]
            for q_local, xy in patch.coords.items():
                coords[off + q_local] = xy
        return coords

    # ----- seams -----------------------------------------------------------

    def register_seam(
        self,
        kind: str,
        a: str,
        b: str,
        pairs: Iterable[Tuple[int, int]],
    ) -> None:
        """Register a seam of type 'rough' or 'smooth' between patches a and b.

        Pairs are local-qubit index pairs (i_in_a, j_in_b). The (a,b) order is
        significant for metadata, but globalization is symmetric.
        """
        k = kind.strip().lower()
        if k not in {"rough", "smooth"}:
            raise ValueError("Seam kind must be 'rough' or 'smooth'")
        if a not in self.patches or b not in self.patches:
            raise KeyError("Both patches must be present in layout before registering a seam")
        key = (k, a, b)
        self.seams[key] = list(pairs)

    # ----- helpers ---------------------------------------------------------

    def _globalize_local_indices(
        self, patch_name: str, locals_list: Iterable[int]
    ) -> List[int]:
        offs = self.offsets()
        base = offs[patch_name]
        return [base + i for i in locals_list]

    def _globalize_local_pauli_string(
        self, patch_name: str, local_str: str
    ) -> Tuple[List[int], List[str]]:
        """Return positions and chars for non-identity entries at global indices."""
        offs = self.offsets()
        base = offs[patch_name]
        positions: List[int] = []
        chars: List[str] = []
        for i, c in enumerate(local_str):
            if c in {"X", "Y", "Z"}:
                positions.append(base + i)
                chars.append(c)
        return positions, chars

    # ----- globalization snapshot -----------------------------------------

    def globalize(self) -> Dict[str, object]:
        """Return a snapshot of concatenated stabilizers and coords (idle sanity)."""
        offs = self.offsets()
        total_n = self.global_n()

        def to_global_strings(kind: str) -> List[str]:
            out: List[str] = []
            for name in self._order:
                patch = self.patches[name]
                locals_list = patch.z_stabs if kind == "Z" else patch.x_stabs
                base = offs[name]
                for s in locals_list:
                    chars = ["I"] * total_n
                    for i, c in enumerate(s):
                        if c != "I":
                            chars[base + i] = c
                    out.append("".join(chars))
            return out

        return {
            "n": total_n,
            "z_stabs": to_global_strings("Z"),
            "x_stabs": to_global_strings("X"),
            "coords": self.global_coords(),
            "offsets": offs,
        }


def create_single_patch_layout(
    model,
    *,
    name: str = "q0",
    offset: int = 0
) -> Layout:
    """Create a single-patch Layout from a code model.
    
    Args:
        model: Code model with attributes: code, z_stabilizers, x_stabilizers, logical_z, logical_x
        name: Name for the patch in the layout
        offset: Starting qubit index offset (usually 0)
        
    Returns:
        Layout containing a single patch ready for use with GlobalStimBuilder
    """
    # Create coordinates for local indices 0..n-1
    coords = {i: (float(i), 0.0) for i in range(model.code.n)}
    
    # Create the patch object
    patch = PatchObject(
        n=model.code.n,
        z_stabs=list(model.z_stabilizers),
        x_stabs=list(model.x_stabilizers),
        logical_z=model.logical_z,
        logical_x=model.logical_x,
        coords=coords,
    )
    
    # Create layout and add the patch
    layout = Layout()
    layout.add_patch(name, patch)
    
    return layout


