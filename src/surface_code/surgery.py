# src/surface_code/surgery.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


from .layout import Layout, BoundaryType
from .stim_builder import PhaseSpec  # or: from . import PhaseSpec, depending on your exports
from .stabilizers import _commuting_boundary_mask
from .logicals import _align_logical_x_to_masked_z, _multiply_paulis_disjoint
from .linalg import rank_gf2
from .stabilizers import stabs_to_symplectic




@dataclass
class CNOTSpec:
    """Container for a lattice-surgery CNOT construction.

    phases:
        Ordered list of spacetime phases (pre-merge, merges/splits, post-merge).
    logical_z_control:
        Embedded logical-Z Pauli string for the control patch (global indices).
    logical_x_target:
        Embedded logical-X Pauli string for the target patch (global indices).
    """

    phases: List[PhaseSpec]
    logical_z_control: str | None
    logical_x_target: str | None
    patch_logicals: dict[str, dict[str, str]]  # patch -> {'Z': ..., 'X': ...}


class LatticeSurgery:
    """High-level helper for lattice-surgery constructions on a Layout.

    This class wraps the geometric information from `Layout` and the
    single-patch model to produce stabilizer sets for:
      * disjoint multi-patch memory phases,
      * smooth merges along smooth boundaries, and
      * rough merges along rough boundaries.

    It does *not* know about Stim circuits directly; instead it returns
    Pauli-string stabilizer sets (on the combined code) and logical
    observables. You then feed these into `PhenomenologicalStimBuilder`
    via PhaseSpec + run_phases.
    """

    def __init__(self, layout: Layout):
        self.layout = layout
        self.single_model = layout.single_model
        self.n_single = self.single_model.code.n
        self.n_total = layout.n_total

        # Single-patch stabilizer data
        self.z_single = list(self.single_model.z_stabilizers)
        self.x_single = list(self.single_model.x_stabilizers)
        # Single-patch logicals
        self.logical_z_single = self.single_model.logical_z
        self.logical_x_single = self.single_model.logical_x

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def _seam_idle_stabilizers(
        self, skip_seam: tuple[str, str] | None = None
    ) -> tuple[list[str], list[str]]:
        """Return stabilizers that pin seam ancillas when they are idle.

        We pin each seam ancilla with a single-qubit Z check. The optional
        `skip_seam` lets us omit the seam currently used in a merge window so
        those ancillas can participate in joint checks instead.
        """
        z_pins: list[str] = []
        x_pins: list[str] = []

        def _is_skip(seam: tuple[str, str]) -> bool:
            if skip_seam is None:
                return False
            a, b = seam
            sa, sb = skip_seam
            return (a == sa and b == sb) or (a == sb and b == sa)

        for seam, ancillas in self.layout.seam_qubits.items():
            if _is_skip(seam):
                continue
            for q in ancillas:
                chars = ["I"] * self.n_total
                chars[q] = "Z"
                z_pins.append("".join(chars))

        return z_pins, x_pins

    def _phase_k(self, z_stabs: list[str], x_stabs: list[str]) -> int:
        S = stabs_to_symplectic(z_stabs, x_stabs)
        r = rank_gf2(S)
        return self.n_total - r

    def _embed_patch(self, pauli_str: str, patch_name: str) -> str:
        """Embed a single-patch Pauli into the global index space at patch_name."""
        assert len(pauli_str) == self.n_single
        offset = self.layout.patch_offsets[patch_name]
        left = "I" * offset
        mid = pauli_str
        right = "I" * (self.n_total - offset - self.n_single)
        return left + mid + right

    def _base_stabilizers(self, patches: Sequence[str] | None = None) -> Tuple[List[str], List[str]]:
        """Disjoint-memory stabilizers: same single-patch model copied to each patch."""
        patch_list = list(patches) if patches is not None else list(self.layout.patch_order)
        base_z: List[str] = []
        base_x: List[str] = []

        for s in self.z_single:
            for name in patch_list:
                base_z.append(self._embed_patch(s, name))

        for s in self.x_single:
            for name in patch_list:
                base_x.append(self._embed_patch(s, name))

        return base_z, base_x


    # ------------------------------------------------------------------
    # Internal masking helper shared by smooth/rough merges
    # ------------------------------------------------------------------

    def _masked_embedded_for_boundary(
        self,
        *,
        boundary_type: BoundaryType,          # "smooth" or "rough"
        merge_patches: Sequence[str],         # patches whose boundary is being merged
        all_patches: Sequence[str],           # all logical patches present
        verbose: bool = False,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Mask single-patch stabilizers at a given boundary and embed globally.

        Returns:
            (z_out, x_out, masked_z_local)

            z_out / x_out:
                Z/X stabilizers on the *combined* code where patches in
                `merge_patches` use the boundary-masked single-patch model,
                and all other patches use the unmasked model.
            masked_z_local:
                The *single-patch* Z stabilizers after masking (used by
                rough-merge logical-X alignment).
        """
        local_boundary = self.layout.local_boundary_qubits[boundary_type]
        strip_pauli = "X" if boundary_type == "smooth" else "Z"

        masked_z_local, masked_x_local = _commuting_boundary_mask(
            z_stabilizers=self.z_single,
            x_stabilizers=self.x_single,
            boundary=local_boundary,
            strip_pauli=strip_pauli,
            verbose=verbose,
        )

        merge_set = set(merge_patches)
        z_out: List[str] = []
        x_out: List[str] = []

        # Z stabilizers
        for s, s_masked in zip(self.z_single, masked_z_local):
            for name in all_patches:
                if name in merge_set:
                    z_out.append(self._embed_patch(s_masked, name))
                else:
                    z_out.append(self._embed_patch(s, name))

        # X stabilizers
        for s, s_masked in zip(self.x_single, masked_x_local):
            for name in all_patches:
                if name in merge_set:
                    x_out.append(self._embed_patch(s_masked, name))
                else:
                    x_out.append(self._embed_patch(s, name))

        return z_out, x_out, masked_z_local

    # ------------------------------------------------------------------
    # Smooth and rough merge stabilizers
    # ------------------------------------------------------------------

    def _smooth_merge_stabilizers(
        self,
        left: str,
        right: str,
        *,
        all_patches: Sequence[str],
        verbose: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Return stabilizers for a smooth merge between `left` and `right`.

        - Patches in `all_patches` are all logically present (e.g. ["C", "INT", "T"]).
        - `left` and `right` are the two patches being smoothly merged.
        """
        merge_patches = [left, right]
        z_merge, x_merge, _ = self._masked_embedded_for_boundary(
            boundary_type="smooth",
            merge_patches=merge_patches,
            all_patches=all_patches,
            verbose=verbose,
        )

        # Add joint Z checks tying left, seam, and right along the smooth boundary.
        boundary_left = self.layout.boundary_qubits[left]["smooth"]
        boundary_right = self.layout.boundary_qubits[right]["smooth"]
        seam = self.layout.get_seam_qubits(left, right)

        for q_l, q_r, q_sea in zip(boundary_left, boundary_right, seam):
            chars = ["I"] * self.n_total
            chars[q_l] = "Z"
            chars[q_sea] = "Z"
            chars[q_r] = "Z"
            z_merge.append("".join(chars))

        """
        # Explicitly include the logical Z parity Z_L(left) * Z_L(right) as a stabilizer
        if self.logical_z_single is not None:
            z_left = self._embed_patch(self.logical_z_single, left)
            z_right = self._embed_patch(self.logical_z_single, right)
            z_merge.append(_multiply_paulis_disjoint(z_left, z_right))
        """
        return z_merge, x_merge

    def _rough_merge_stabilizers(
        self,
        left: str,
        right: str,
        *,
        all_patches: Sequence[str],
        verbose: bool = False,
    ) -> Tuple[List[str], List[str], str | None]:
        """Return stabilizers for a rough merge between `left` and `right`.

        Returns:
            (z_stabs, x_stabs, logical_x_aligned_single_patch)

            logical_x_aligned_single_patch is a *single-patch* Pauli string,
            adjusted to commute with the masked Z checks used during rough
            merge. You typically embed this at the *target* patch to define
            the logical-X observable.
        """
        merge_patches = [left, right]
        z_merge, x_merge, masked_z_local = self._masked_embedded_for_boundary(
            boundary_type="rough",
            merge_patches=merge_patches,
            all_patches=all_patches,
            verbose=verbose,
        )

        # Adjust logical X to commute with masked Z checks (single-patch space).
        logical_x_aligned: str | None = _align_logical_x_to_masked_z(
            self.single_model.logical_x,
            self.x_single,
            masked_z_local,
            verbose=verbose,
        )

        # Add joint X checks tying left, seam, and right along the rough boundary.
        boundary_left = self.layout.boundary_qubits[left]["rough"]
        boundary_right = self.layout.boundary_qubits[right]["rough"]
        seam = self.layout.get_seam_qubits(left, right)

        for q_l, q_r, q_sea in zip(boundary_left, boundary_right, seam):
            chars = ["I"] * self.n_total
            chars[q_l] = "X"
            chars[q_sea] = "X"
            chars[q_r] = "X"
            x_merge.append("".join(chars))

        """
        # Explicitly include the logical X parity X_L(left) * X_L(right) as a stabilizer
        if logical_x_aligned is not None:
            x_left = self._embed_patch(logical_x_aligned, left)
            x_right = self._embed_patch(logical_x_aligned, right)
            x_merge.append(_multiply_paulis_disjoint(x_left, x_right))
        """

        return z_merge, x_merge, logical_x_aligned


    #------------------------------------------------------------------
    # Extract logical operators for a given patch
    #------------------------------------------------------------------
    def _logical_for_patch(self, patch: str, basis: str) -> str | None:
        """Return embedded logical operator on a given patch and basis.

        basis: 'Z' or 'X'
        Returns a Pauli string on the *combined* code, or None if that logical
        isn't available in the single-patch model.
        """
        basis = basis.upper()
        if basis == "Z":
            local = self.single_model.logical_z
        elif basis == "X":
            local = self.single_model.logical_x
        else:
            raise ValueError("basis must be 'Z' or 'X'")

        if local is None:
            return None

        return self._embed_patch(local, patch)
    # ------------------------------------------------------------------
    # High-level: CNOT via lattice surgery
    # ------------------------------------------------------------------

    def cnot(
        self,
        control: str,
        ancilla: str,
        target: str,
        *,
        rounds_pre: int,
        rounds_merge: int,
        rounds_post: int,
        verbose: bool = False,
    ) -> CNOTSpec:
        """Construct a CNOT via smooth(control–ancilla) + rough(ancilla–target).

        Returns:
            CNOTSpec with:
              * phases: a PhaseSpec list suitable for builder.run_phases(...)
              * logical_z_control: embedded logical-Z on `control`
              * logical_x_target: embedded aligned logical-X on `target`
        """
        # Which logical patches exist in this 3-patch CNOT layout
        patches = [control, ancilla, target]


        # Precompute both Z/X logicals for each patch, embedded globally.
        patch_logicals: dict[str, dict[str, str]] = {}
        for name in patches:
            patch_logicals[name] = {}
            z_emb = self._logical_for_patch(name, "Z")
            x_emb = self._logical_for_patch(name, "X")
            if z_emb is not None:
                patch_logicals[name]["Z"] = z_emb
            if x_emb is not None:
                patch_logicals[name]["X"] = x_emb



        # Disjoint 3-patch memory stabilizers
        base_z, base_x = self._base_stabilizers(patches)
        # Pin seam ancillas during idle phases so they don't contribute spurious k.
        all_seam_z, all_seam_x = self._seam_idle_stabilizers()

        # C+INT smooth merge
        smooth_merge_z, smooth_merge_x = self._smooth_merge_stabilizers(
            control,
            ancilla,
            all_patches=patches,
            verbose=verbose,
        )
        # Only INT–T seam is idle during the smooth merge, so pin that one.
        smooth_merge_z += self._seam_idle_stabilizers(skip_seam=(control, ancilla))[0]

        # INT+T rough merge
        rough_merge_z, rough_merge_x, logical_x_aligned = self._rough_merge_stabilizers(
            ancilla,
            target,
            all_patches=patches,
            verbose=verbose,
        )
        # Only C–INT seam is idle during the rough merge, so pin that one.
        rough_merge_z += self._seam_idle_stabilizers(skip_seam=(ancilla, target))[0]

        # If we needed to align logical X for commutation, use the aligned version everywhere
        # (so init and final logicals stay in the same Pauli class).
        if logical_x_aligned is not None:
            patch_logicals[target]["X"] = self._embed_patch(logical_x_aligned, target)

        # Build phase list
        phases: List[PhaseSpec] = [
            PhaseSpec("pre-merge", base_z + all_seam_z,
                base_x + all_seam_x,
                rounds_pre),
            PhaseSpec(
                f"{control}+{ancilla} smooth merge",
                smooth_merge_z,
                smooth_merge_x,
                rounds_merge,
                measure_z=True,
                measure_x=True,
            ),
            PhaseSpec(
                f"{control}|{ancilla} smooth split",
                base_z + all_seam_z,
                base_x + all_seam_x,
                rounds_merge,
            ),
            PhaseSpec(
                f"{ancilla}+{target} rough merge",
                rough_merge_z,
                rough_merge_x,
                rounds_merge,
                measure_z=True,
                measure_x=True,
            ),
            PhaseSpec(
                f"{ancilla}|{target} rough split",
                base_z + all_seam_z,
                base_x + all_seam_x,
                rounds_merge,
            ),
            PhaseSpec("post-merge", base_z + all_seam_z,
                base_x + all_seam_x,
                rounds_post),
        ]

        # Embedded logical observables
        logical_z_control: str | None = None
        if self.single_model.logical_z is not None:
            logical_z_control = self._embed_patch(self.single_model.logical_z, control)

        logical_x_target: str | None = None
        if logical_x_aligned is not None:
            logical_x_target = self._embed_patch(logical_x_aligned, target)


        if verbose:
            print("[debug] code parameters per phase:")
            for ph in phases:
                k = self._phase_k(ph.z_stabilizers, ph.x_stabilizers)
                print(f"  Phase {ph.name:24s}: n={self.n_total}, k={k}, "
                    f"#Z={len(ph.z_stabilizers)}, #X={len(ph.x_stabilizers)}")


        return CNOTSpec(
            phases=phases,
            logical_z_control=logical_z_control,
            logical_x_target=logical_x_target,
            patch_logicals=patch_logicals,
        )
