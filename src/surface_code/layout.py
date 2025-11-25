from dataclasses import dataclass
from typing import Literal, Dict, List, Tuple


from . import build_surface_code_model
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

    # --- internal helpers ---
    def _calculate_boundaries(self) -> None:
        smooth = find_smooth_boundary_data_qubits(self.single_model)
        rough = find_rough_boundary_data_qubits(self.single_model)
        self._local_smooth = smooth
        self._local_rough = rough

    def _calculate_layout(self) -> None:
        n_single = self.single_model.code.n

        # map patch -> offset
        patch_offsets: Dict[str, int] = {}
        seam_qubits: Dict[Tuple[str, str], List[int]] = {}

        offset = 0
        # first place all patches in order
        for name in self.patch_order:
            patch_offsets[name] = offset
            offset += n_single

        # then insert seam ancillas in between patches
        # (you can choose a convention for where they sit linearly;
        #  here we just append them in the order seams are given)
        for seam in self.seams:
            if seam.boundary_type == "smooth":
                local_boundary = self._local_smooth
            else:
                local_boundary = self._local_rough

            ancilla_indices = list(range(offset, offset + len(local_boundary)))
            seam_qubits[(seam.left, seam.right)] = ancilla_indices
            offset += len(local_boundary)

        self._n_total = offset
        self._patch_offsets = patch_offsets

        # precompute global boundary qubits
        boundary_qubits: Dict[str, Dict[BoundaryType, List[int]]] = {}
        for name, off in patch_offsets.items():
            boundary_qubits[name] = {
                "smooth": [off + q for q in self._local_smooth],
                "rough": [off + q for q in self._local_rough],
            }
        self._boundary_qubits = boundary_qubits
        self._seam_qubits = seam_qubits

    # convenience accessors
    def get_patch_offset(self, patch_name: str) -> int:
        return self._patch_offsets[patch_name]

    def get_seam_qubits(self, patch1: str, patch2: str) -> List[int]:
        key = (patch1, patch2)
        if key in self._seam_qubits:
            return self._seam_qubits[key]
        key_rev = (patch2, patch1)
        return self._seam_qubits[key_rev]

    def get_boundary_qubits(self, patch_name: str, boundary_type: BoundaryType) -> List[int]:
        return self._boundary_qubits[patch_name][boundary_type]