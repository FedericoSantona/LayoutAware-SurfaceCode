from dataclasses import dataclass
from typing import Literal, Dict, List, Tuple

from .geometry_utils import (
    find_smooth_boundary_data_qubits,
    find_rough_boundary_data_qubits,
)

BoundaryType = Literal["smooth", "rough"]

@dataclass(frozen=True)
class SeamSpec:
    left: str
    right: str
    boundary_type: BoundaryType

class Layout:
    def __init__(
        self,
        distance: int,
        code_type: str,
        patch_order: List[str],
        seams: List[SeamSpec],
        patch_metadata: Dict[str, str] | None = None,
    ):
        # Lazy import to avoid circular dependency with __init__.py
        from . import build_surface_code_model
        self.single_model = build_surface_code_model(distance, code_type)
        self.patch_order = patch_order
        self.seams = seams
        self.patch_metadata = patch_metadata or {}

        self._calculate_boundaries()
        self._calculate_layout()

    # --- public properties ---
    @property
    def n_total(self) -> int:
        return self._n_total

    @property
    def patch_offsets(self) -> Dict[str, int]:
        return self._patch_offsets

    @property
    def seam_qubits(self) -> Dict[Tuple[str, str], List[int]]:
        return self._seam_qubits

    @property
    def boundary_qubits(self) -> Dict[str, Dict[BoundaryType, List[int]]]:
        return self._boundary_qubits

    @property
    def local_boundary_qubits(self) -> Dict[BoundaryType, List[int]]:
        return {
            "smooth": list(self._local_smooth),
            "rough": list(self._local_rough),
        }

    # --- internal helpers ---
    def _calculate_boundaries(self) -> None:
        smooth = find_smooth_boundary_data_qubits(self.single_model)
        rough = find_rough_boundary_data_qubits(self.single_model)
        self._local_smooth = smooth
        self._local_rough = rough

    def _calculate_layout(self) -> None:
        """Compute global qubit indices for patches and seams.

        Patches are placed in the order given by `self.patch_order`.
        For each adjacent pair (P_i, P_{i+1}), if there is a SeamSpec
        connecting them, we insert a seam block of ancilla qubits
        *between* the two patches. The size of the seam block is
        determined by the number of boundary data qubits of the
        appropriate type (smooth or rough) on the single-patch model.
        """
        n_single = self.single_model.code.n

        # map patch -> offset
        patch_offsets: Dict[str, int] = {}
        # map (left, right) -> list of ancilla indices
        seam_qubits: Dict[Tuple[str, str], List[int]] = {}

        offset = 0

        # Convenience: index seams by the unordered pair of endpoints
        # so we can look up a seam for (prev, curr) regardless of left/right
        seams_by_pair: Dict[frozenset[str], SeamSpec] = {}
        for seam in self.seams:
            pair_key = frozenset({seam.left, seam.right})
            seams_by_pair[pair_key] = seam

        # Walk along the patch order, inserting patches and seams
        for i, name in enumerate(self.patch_order):
            # Place the patch
            patch_offsets[name] = offset
            offset += n_single

            # If there is a "next" patch, see if there's a seam between them
            if i < len(self.patch_order) - 1:
                next_name = self.patch_order[i + 1]
                pair_key = frozenset({name, next_name})
                seam = seams_by_pair.get(pair_key, None)

                if seam is not None:
                    # Choose local boundary type for this seam
                    if seam.boundary_type == "smooth":
                        local_boundary = self._local_smooth
                    else:
                        local_boundary = self._local_rough

                    # Allocate ancillas for this seam
                    ancilla_indices = list(
                        range(offset, offset + len(local_boundary))
                    )

                    # Store using the original (left, right) orientation
                    seam_qubits[(seam.left, seam.right)] = ancilla_indices
                    offset += len(local_boundary)

        # Total number of qubits in the combined layout
        self._n_total = offset
        self._patch_offsets = patch_offsets

        # Precompute global boundary qubits for each patch
        boundary_qubits: Dict[str, Dict[BoundaryType, List[int]]] = {}
        for name, off in patch_offsets.items():
            boundary_qubits[name] = {
                "smooth": [off + q for q in self._local_smooth],
                "rough": [off + q for q in self._local_rough],
            }

        self._boundary_qubits = boundary_qubits
        self._seam_qubits = seam_qubits

    def get_seam_qubits(self, patch1: str, patch2: str) -> List[int]:
        key = (patch1, patch2)
        if key in self._seam_qubits:
            return self._seam_qubits[key]
        key_rev = (patch2, patch1)
        return self._seam_qubits[key_rev]

    def print_layout(self) -> None:
        """Pretty-print a summary of the combined layout.

        This shows:
          * qubit ranges for each patch and seam in 1D index space
          * total vs expected qubit count
          * boundary qubits in global coordinates for smooth/rough boundaries
        """
        n_single = self.single_model.code.n

        print("\n" + "=" * 60)
        print("Qubit Layout Verification:")
        print("=" * 60)

        # Reconstruct the same adjacency we used when building the layout,
        # so we can interleave patches and seams in order.
        seams_by_pair: Dict[frozenset[str], SeamSpec] = {}
        for seam in self.seams:
            seams_by_pair[frozenset({seam.left, seam.right})] = seam

        # Print patches and any seams between adjacent patches
        for i, name in enumerate(self.patch_order):
            offset = self.patch_offsets[name]
            print(f"Patch {name:<6s} qubits [{offset:4d}, {offset + n_single:4d})")

            if i < len(self.patch_order) - 1:
                next_name = self.patch_order[i + 1]
                pair_key = frozenset({name, next_name})
                seam = seams_by_pair.get(pair_key)
                if seam is not None:
                    # Find the ancilla indices for this seam
                    ancillas = self.seam_qubits.get((seam.left, seam.right))
                    if ancillas is None:
                        ancillas = self.seam_qubits.get((seam.right, seam.left), [])
                    if ancillas:
                        start = ancillas[0]
                        end = ancillas[-1] + 1
                        seam_label = f"{seam.left}_{seam.right}"
                        print(f"Seam {seam_label:<6s} qubits [{start:4d}, {end:4d})")

        # Totals
        print(f"Total qubits:   {self.n_total}")
        expected_total = len(self.patch_order) * n_single + sum(
            len(self._local_smooth if seam.boundary_type == "smooth" else self._local_rough)
            for seam in self.seams
        )
        print(f"Expected total: {expected_total}")

        # Boundary qubits in global coordinates
        print("\nBoundary Qubits (in global coordinates):")
        for btype in ("smooth", "rough"):
            print(f"  {btype.capitalize()} boundary:")
            for name in self.patch_order:
                bqs = self.boundary_qubits[name][btype]  # already global
                print(f"    Patch {name:<6s} {bqs}")

        print("=" * 60 + "\n")
