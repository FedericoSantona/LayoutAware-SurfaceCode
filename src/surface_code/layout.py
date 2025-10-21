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

    @classmethod
    def from_code_model(cls, model, *, name: str = "q0") -> "PatchObject":
        """Create a PatchObject from a heavy-hex code model.
        
        Args:
            model: HeavyHexModel with code attribute
            name: Name for the patch (unused but kept for compatibility)
            
        Returns:
            PatchObject with proper 2D coordinates extracted from geometry
        """
        # Extract actual 2D coordinates from heavy-hex geometry
        try:
            raw_coords = cls._extract_qubit_coordinates(model.code)
            # Convert numpy arrays to tuples for compatibility
            coords = {i: tuple(pos) for i, pos in raw_coords.items()}
        except RuntimeError:
            # Fallback: generate a reasonable 2D grid if geometry not available
            raise RuntimeError("Failed to extract qubit coordinates from heavy-hex geometry")
        return cls(n=model.code.n, z_stabs=list(model.z_stabilizers), x_stabs=list(model.x_stabilizers), logical_z=model.logical_z, logical_x=model.logical_x, coords=coords)

    @staticmethod
    def _extract_qubit_coordinates(code):
        """Extract actual 2D coordinates from heavy-hex code geometry.
        
        Args:
            code: StabSubSystemCode object built via geometry-backed builder
            
        Returns:
            dict: {logical_qubit_index: (x, y)} mapping logical qubit indices to 2D coordinates
        """
        import numpy as np
        
        # tolerate minor API differences
        shell = getattr(code, "shell", None) or getattr(code, "_shell", None)
        qd = getattr(code, "qubit_data", None) or getattr(code, "_qubit_data", None)
        
        if shell is None or qd is None:
            raise RuntimeError("This code object has no geometry. Build it via a geometry-backed builder.")
        
        # Get the mapping from physical qubit IDs to logical indices
        qubit_to_index = getattr(qd, "qubit_to_index", None)
        if qubit_to_index is None:
            raise RuntimeError("No qubit_to_index mapping found in qubit_data")
        
        coords = {}
        for v in shell.vertices:
            physical_qubit_id = qd.qubit[v.id]  # vertex-id -> physical qubit ID
            logical_index = qubit_to_index[physical_qubit_id]  # physical ID -> logical index
            
            # vertices expose a position; some builds use .pos, others .position (property)
            pos = getattr(v, "pos", None)
            if pos is None:
                pos = v.position
            # first hit wins; vertices sharing a qubit will agree on position
            coords.setdefault(logical_index, np.array(pos, dtype=float))

        print("the qubit coordinates are: ", coords)
        
        return coords  # dict: {logical_index: np.array([x, y])}

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

    def globalize_local_pauli_string(
        self, patch_name: str, local_str: str
    ) -> Tuple[List[int], List[str]]:
        """Convert local Pauli string to global positions and characters.
        
        Returns:
            Tuple of (global_positions, pauli_chars) for non-identity entries
        """
        offs = self.offsets()
        base = offs[patch_name]
        positions: List[int] = []
        chars: List[str] = []
        for i, c in enumerate(local_str):
            if c in {"X", "Y", "Z"}:
                positions.append(base + i)
                chars.append(c)
        return positions, chars

    def globalize_local_index(self, patch_name: str, local_index: int) -> int:
        """Convert a single local qubit index to global index."""
        offs = self.offsets()
        base = offs[patch_name]
        return base + local_index

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
                for s in locals_list:
                    chars = ["I"] * total_n
                    positions, pauli_chars = self.globalize_local_pauli_string(name, s)
                    for pos, char in zip(positions, pauli_chars):
                        chars[pos] = char
                    out.append("".join(chars))
            return out

        return {
            "n": total_n,
            "z_stabs": to_global_strings("Z"),
            "x_stabs": to_global_strings("X"),
            "coords": self.global_coords(),
            "offsets": offs,
        }

    def plot(
    self,
    *,
    annotate: bool = False,
    seams: bool = True,
    figsize: Tuple[float, float] = (7, 7),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    ax=None,
    seam_markers: bool = True,
    seam_labels: bool = False,
    label_fontsize: int = 8,
    ) -> None:
        """Visualize the global qubit layout and (optionally) seam connections.

        Args:
            annotate: If True, label global qubit indices.
            seams: If True, draw rough/smooth seam connections between patches.
            figsize: Matplotlib figure size when creating a new figure.
            title: Optional title for the plot.
            save_path: If provided, saves the figure to this path.
            ax: Optional matplotlib axis to draw on. If None, creates a new figure.
            seam_markers: If True, draw markers at seam endpoints (square for rough, triangle for smooth).
            seam_labels: If True, label each seam at its midpoint with an index and type (e.g., R0, S3).
            label_fontsize: Font size for annotations and seam labels.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.lines as mlines
        except Exception as exc:
            raise RuntimeError(
                "matplotlib is required for layout visualization. Please install it."
            ) from exc

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        coords = self.global_coords()
        offs = self.offsets()

        # Map global index -> patch name
        global_to_patch: Dict[int, str] = {}
        for name in self._order:
            base = offs[name]
            nloc = self.patches[name].n
            for i in range(nloc):
                global_to_patch[base + i] = name

        # Collect per-patch point clouds
        per_patch_points: Dict[str, List[Tuple[float, float, int]]] = {
            name: [] for name in self._order
        }
        for g_idx, (x, y) in coords.items():
            pname = global_to_patch.get(g_idx, "?")
            per_patch_points[pname].append((x, y, g_idx))

        # Prepare to collect patch handles for legend
        patch_handles = []
        # Scatter per patch
        for pname in self._order:
            pts = per_patch_points[pname]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            h = ax.scatter(xs, ys, label=f"patch: {pname}")
            patch_handles.append(h)
            if annotate:
                for x, y, g in pts:
                    ax.text(x, y, str(g), fontsize=label_fontsize, ha="center", va="center")

        # Draw seams
        if seams and self.seams:
            kinds_present = set()
            pair_counters = {"rough": 0, "smooth": 0}
            for (kind, a, b), pairs in self.seams.items():
                kinds_present.add(kind)
                color = "tab:red" if kind == "rough" else "tab:blue"
                ls = "--" if kind == "rough" else "-"
                marker = "s" if kind == "rough" else "^"  # square vs triangle
                a_off = offs[a]
                b_off = offs[b]
                for (i_local, j_local) in pairs:
                    gi = a_off + i_local
                    gj = b_off + j_local
                    if gi in coords and gj in coords:
                        (x1, y1) = coords[gi]
                        (x2, y2) = coords[gj]
                        # seam line
                        ax.plot(
                            [x1, x2], [y1, y2],
                            linestyle=ls, color=color, linewidth=2, alpha=0.9, zorder=2
                        )
                        # endpoint markers
                        if seam_markers:
                            ax.scatter(
                                [x1, x2], [y1, y2],
                                marker=marker, s=36,
                                edgecolors="black", facecolors=color, zorder=3
                            )
                        # midpoint label
                        if seam_labels:
                            midx = 0.5 * (x1 + x2)
                            midy = 0.5 * (y1 + y2)
                            tag = ("R" if kind == "rough" else "S") + str(pair_counters[kind])
                            ax.text(
                                midx, midy, tag,
                                fontsize=label_fontsize, ha="center", va="center",
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.8),
                                zorder=4,
                            )
                        pair_counters[kind] += 1

            # Legend entries listing each seam and its type
            handles = []
            for (k_kind, k_a, k_b) in self.seams.keys():
                color_k = "tab:red" if k_kind == "rough" else "tab:blue"
                ls_k = "--" if k_kind == "rough" else "-"
                label_k = f"{k_kind}: {k_a}↔{k_b}"
                handles.append(mlines.Line2D([], [], color=color_k, linestyle=ls_k, label=label_k))
            if handles or patch_handles:
                combined = patch_handles + handles
                ax.legend(handles=combined, loc="best", title="Patches and seams")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        if title:
            ax.set_title(title)

        if save_path:
            ax.figure.savefig(save_path, bbox_inches="tight")
        


def create_single_patch_layout(
    model,
    *,
    name: str = "q0",
) -> Layout:
    """Create a single-patch Layout from a code model.
    
    Args:
        model: Code model with attributes: code, z_stabilizers, x_stabilizers, logical_z, logical_x
        name: Name for the patch in the layout
        offset: Starting qubit index offset (usually 0)
        
    Returns:
        Layout containing a single patch ready for use with GlobalStimBuilder
    """
    # Create the patch object using the class method
    patch = PatchObject.from_code_model(model, name=name)
    
    # Create layout and add the patch
    layout = Layout()
    layout.add_patch(name, patch)
    
    return layout


