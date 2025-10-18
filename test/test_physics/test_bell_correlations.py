import pytest
import numpy as np

from qiskit import QuantumCircuit

from surface_code import build_heavy_hex_model, PhenomenologicalStimConfig
from surface_code.layout import PatchObject
from surface_code.surgery_compile import compile_circuit_to_surgery
from surface_code.builder import GlobalStimBuilder


@pytest.mark.slow
@pytest.mark.parametrize("shots", [20000])
@pytest.mark.parametrize("px,pz", [(0.0, 0.0), (5e-3, 5e-3)])
def test_bell_correlations_joint_mpps(shots, px, pz):
    # Build simple Bell circuit H-CX
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Build code model
    d = 3
    model = build_heavy_hex_model(d)

    # Build one patch per logical qubit
    def build_patch():
        return PatchObject(
            n=model.code.n,
            z_stabs=model.z_stabilizers,
            x_stabs=model.x_stabilizers,
            logical_z=model.logical_z,
            logical_x=model.logical_x,
            coords={i: (float(i), 0.0) for i in range(model.code.n)},
        )

    patches = {f"q{i}": build_patch() for i in range(qc.num_qubits)}

    # Simple seams: pair (i,i)
    default_pairs = [(i, i) for i in range(d)]
    seams = {("rough", "q0", "q1"): list(default_pairs), ("smooth", "q0", "q1"): list(default_pairs)}

    # Bracket in Z; demo both bases
    bracket_map = {"q0": "Z", "q1": "Z"}
    layout, ops = compile_circuit_to_surgery(qc, patches, seams, distance=d, bracket_map=bracket_map, warmup_rounds=1)

    stim_cfg = PhenomenologicalStimConfig(
        rounds=d,
        p_x_error=float(px),
        p_z_error=float(pz),
        init_label="0",
        bracket_basis="Z",
        demo_basis=["Z", "X"],
        demo_joint_only=False,
    )

    gb = GlobalStimBuilder(layout)
    circuit, observable_pairs, metadata = gb.build(ops, stim_cfg, bracket_map, qc)

    # Ensure ordering tail contains joint then singles for each basis
    tail = str(circuit).strip().splitlines()[-80:]
    # Expect at least one joint MPP present
    assert any(line.strip().startswith("MPP ") for line in tail)

    # Sample measurement record directly for joint demos
    sampler = circuit.compile_sampler(seed=123)
    m = sampler.sample(shots=int(shots))

    joint_meta = metadata.get("joint_demos", {})
    assert joint_meta, "joint_demos metadata missing"

    # Collect ZZ and XX indices for pair (q0,q1)
    zz_bits = None
    xx_bits = None
    for k, info in joint_meta.items():
        idx = info.get("index")
        basis = info.get("basis")
        pair = tuple(info.get("pair", []))
        if idx is None or pair != ("q0", "q1"):
            continue
        col = np.asarray(m[:, int(idx)], dtype=np.uint8)
        if basis == "Z":
            zz_bits = col
        elif basis == "X":
            xx_bits = col

    assert zz_bits is not None and xx_bits is not None, "Missing joint ZZ/XX bits"

    # Compute expectations E = 1 - 2 p1
    zz = 1.0 - 2.0 * float(zz_bits.mean())
    xx = 1.0 - 2.0 * float(xx_bits.mean())

    if px == 0.0 and pz == 0.0:
        assert zz > 0.95 and xx > 0.95
    else:
        # With noise, correlations should remain positive
        assert zz > 0.0 and xx > 0.0
