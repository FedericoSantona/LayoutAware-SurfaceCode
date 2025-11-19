"""Quick playground for the heavy-hex transpilation pipeline.

This script stays intentionally thin: it wires together the high-level
components already defined in the project so you can iterate on new QEC
experiments without re-implementing logic. Extend it as you develop new
benchmarks, targets, or analysis steps.
"""

from __future__ import annotations

import os
import sys
import json
import math
import io
import contextlib
import stim
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the project root to Python path to enable src imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure the src/ directory is available for imports when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Import CLI utilities
from cli import parse_args, instantiate_benchmark, load_target, build_config, format_leaderboard

# Import simulation module
from surface_code.simulation import run_logical_simulation

from benchmarks.BenchmarkCircuit import BenchmarkCircuit
from surface_code.utils import (
    plot_heavy_hex_code,
    diagnostic_print,
)
from surface_code.reporting import (
    print_header,
    print_per_qubit_results,
    print_physics_demo,
    print_pauli_frame_audit,
    print_debug_details,
    _print_demo_preamble,
    generate_detailed_json,
    save_detailed_json,
    print_final_state_distribution,
    print_decoded_logical_distribution,
    print_schrodinger_snapshot_distribution,
)

from transpile.pipeline import HeavyHexTranspiler
from surface_code import (
    build_surface_code_model,
    PhenomenologicalStimConfig,
)
from surface_code.layout import PatchObject
from surface_code.surgery_compile import compile_circuit_to_surgery
from surface_code.builder import GlobalStimBuilder
from surface_code.memory_layout import build_memory_layout
from surface_code.dem_utils import (
    circuit_to_graphlike_dem,
    compute_dem_components,
    component_anchor_coverage,
    dem_error_block_histogram,
    iter_error_blocks_with_prob,
    add_spatial_correlations_to_dem,
    add_boundary_hooks_to_dem,
    enforce_component_boundaries,
)

from surface_code.pauli import PauliTracker, parse_init_label, sequence_from_qc


def build_stim_memory_circuit(
    *,
    distance: int,
    rounds: int,
    px: float,
    pz: float,
    p_meas: float,
    init_label: str,
) -> Tuple[stim.Circuit, List[Tuple[Optional[int], Optional[int]]], Dict[str, object]]:
    """Return a Stim-generated rotated-memory circuit plus minimal metadata."""
    init = (init_label or "0").strip()
    init_basis = "Z" if init in {"0", "1"} else "X"
    task = "surface_code:rotated_memory_z" if init_basis == "Z" else "surface_code:rotated_memory_x"
    data_prob = float(max(px, pz))
    circuit = stim.Circuit.generated(
        task,
        distance=int(distance),
        rounds=int(rounds),
        before_round_data_depolarization=data_prob,
        before_measure_flip_probability=float(p_meas),
        after_reset_flip_probability=float(p_meas),
    )
    metadata: Dict[str, object] = {
        "observable_basis": (init_basis,),
        "observable_patches": ("q0",),
        "boundary_anchors": {"detector_ids": []},
        "mwpm_debug": {},
        "cnot_operations": [],
        "demo": {},
        "joint_demos": {},
        "final_snapshot": {},
        "byproducts": [],
        "explicit_logical_brackets": False,
        "noise_model": {
            "p_x_error": float(px),
            "p_z_error": float(pz),
            "p_meas": float(p_meas),
        },
        "boundary_rows": {
            "q0": {
                "Z": {"rows": [], "total": distance},
                "X": {"rows": [], "total": distance},
            }
        },
        "stim_generator": {
            "task": task,
            "distance": distance,
            "rounds": rounds,
        },
    }
    observable_pairs = [(None, None)]
    return circuit, observable_pairs, metadata

def _print_dem_health_report(dem, metadata, project_root: Path, benchmark_name: str) -> None:
    """Pretty, verbose health report for a DEM used in threshold/scaling studies.

    This groups together all existing diagnostics (metadata coverage, temporal
    tag stats, cross-row edges, measurement vs data-like terms) and adds:
      * error-term arity histogram (graphlike vs hypergraph)
      * connected-component and boundary coverage check
      * per-row cross-row degrees
      * probability sanity for measurement-like and data-like terms
    """
    import re
    from collections import Counter

    print("\n[DEM-REPORT] ===== Detector Error Model Health Check (pre-decoder) =====")

    # 1) Dump the DEM text for manual inspection
    dem_text = str(dem)
    plots_dir = project_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    (plots_dir / f"dem_{benchmark_name}.dem.txt").write_text(dem_text)

    # 2) Metadata / detector_context coverage
    dbg = (metadata.get("mwpm_debug", {}) or {})
    det_ctx = dbg.get("detector_context", {}) or {}  # {det_id: {"tag": "...", "context": {...}}}
    tag_stats = dbg.get("tag_stats", {}) or {}

    missing_ctx = {"x_temporal": 0, "z_temporal": 0}
    rows_seen = {"x_temporal": set(), "z_temporal": set()}
    wrong_keys = []
    for did, meta in det_ctx.items():
        tag = (meta or {}).get("tag")
        if tag in ("x_temporal", "z_temporal"):
            ctx = (meta or {}).get("context", {}) or {}
            if "patch" not in ctx or "row" not in ctx:
                missing_ctx[tag] += 1
                wrong_keys.append((did, tag, sorted(ctx.keys())))
            else:
                rows_seen[tag].add((ctx.get("patch"), ctx.get("row")))

    print("[DEM-REPORT] Metadata coverage:")
    print(f"  temporal detectors missing (patch,row): {missing_ctx}")
    print(
        "  unique (patch,row) per temporal tag:",
        {t: len(s) for t, s in rows_seen.items()},
    )
    if wrong_keys[:5]:
        print("  sample detectors with missing/renamed context keys (up to 5):")
        for did, tag, keys in wrong_keys[:5]:
            print(f"    D{did} tag={tag} context_keys={keys}")

    if tag_stats:
        print("[DEM-REPORT] Detector tag statistics (temporal and butterfly):")
        for tag, stats in tag_stats.items():
            if "temporal" in tag or "butterfly" in tag:
                print(
                    f"  {tag}: emitted={stats.get('emitted', 0)}, "
                    f"kept={stats.get('kept', 0)}, dropped={stats.get('dropped', 0)}"
                )

    def _info(did: int):
        meta = det_ctx.get(int(did), {}) or {}
        tag = meta.get("tag")
        ctx = meta.get("context", {}) or {}
        patch = ctx.get("patch")
        row = ctx.get("row")
        return tag, patch, row

    # 3) Parse ERROR components via Stim's API.
    one_det = []  # [(p, d0)]
    multi_det_terms = []  # [(p, [d0, d1, ...])]
    arity_hist = dem_error_block_histogram(dem)
    for p_val, det_list in iter_error_blocks_with_prob(dem):
        if not det_list:
            continue
        if len(det_list) == 1:
            one_det.append((p_val, det_list[0]))
        else:
            multi_det_terms.append((p_val, det_list))

    total_terms = sum(arity_hist.values())
    print("\n[DEM-REPORT] Error-term structure:")
    print(f"  total ERROR terms parsed: {total_terms}")
    print(
        f"  1-detector terms: {len(one_det)}  "
        f"multi-detector terms: {len(multi_det_terms)}"
    )
    print(
        "  arity histogram (detector count -> #terms):",
        dict(sorted(arity_hist.items())),
    )
    max_arity = max(arity_hist.keys()) if arity_hist else 0
    if max_arity > 2:
        print("  WARNING: non-graphlike DEM detected (some terms have >2 detectors).")

    # 4) Connectivity & boundary coverage
    print("\n[DEM-REPORT] Connectivity & boundaries:")
    components, boundary_nodes = compute_dem_components(dem)
    anchor_meta = metadata.get("boundary_anchors", {}) or {}
    explicit_ids = anchor_meta.get("detector_ids") or []
    coverage = component_anchor_coverage(components, boundary_nodes, explicit_ids)
    comp_sizes = [len(comp) for comp in components if comp]

    if components:
        print(
            f"  connected components: {len(components)}  "
            f"(boundaryless={coverage.get('uncovered', 0)})"
        )
        print(
            f"  component sizes: min={min(comp_sizes)} max={max(comp_sizes)} "
            f"avg={sum(comp_sizes) / len(comp_sizes):.1f}"
        )
        uncovered = coverage.get("uncovered_indices") or []
        if uncovered:
            print(f"  uncovered component indices (up to 8): {uncovered[:8]}")
    else:
        print("  no detector nodes found in DEM graph.")

    # 5) Cross-row temporal data-fault candidates (x/z_temporal)
    print("\n[DEM-REPORT] Cross-row temporal data-fault candidates:")
    candidates = []  # (tag, patch, (row_i,row_j), p_val, d0, d1)
    for p_val, ids in multi_det_terms:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d0, d1 = ids[i], ids[j]
                tag0, patch0, row0 = _info(d0)
                tag1, patch1, row1 = _info(d1)
                # same patch, same temporal tag, different rows
                if (
                    tag0 in ("z_temporal", "x_temporal")
                    and tag0 == tag1
                    and patch0 is not None
                    and patch0 == patch1
                    and isinstance(row0, int)
                    and isinstance(row1, int)
                    and row0 != row1
                ):
                    r_pair = tuple(sorted((row0, row1)))
                    candidates.append((tag0, patch0, r_pair, p_val, d0, d1))

    x_candidates = [c for c in candidates if c[0] == "x_temporal"]
    z_candidates = [c for c in candidates if c[0] == "z_temporal"]

    print(f"  x_temporal cross-row pairs: {len(x_candidates)}")
    print(f"  z_temporal cross-row pairs: {len(z_candidates)}")

    if x_candidates:
        print("  sample x_temporal edges (up to 5):")
        for tag, patch, rpair, p_val, d0, d1 in x_candidates[:5]:
            print(f"    {patch} rows={rpair} error({p_val}) D{d0} D{d1}")
    if z_candidates:
        print("  sample z_temporal edges (up to 5):")
        for tag, patch, rpair, p_val, d0, d1 in z_candidates[:5]:
            print(f"    {patch} rows={rpair} error({p_val}) D{d0} D{d1}")
    if not x_candidates and not z_candidates:
        print("  no temporal cross-row candidates found.")

    # 6) Per-row cross-row degrees
    if candidates:
        row_deg = Counter()
        for tag, patch, rpair, _p, _d0, _d1 in candidates:
            for r in rpair:
                row_deg[(tag, patch, r)] += 1
        print("\n[DEM-REPORT] Per-row cross-row degree (top 10 rows):")
        for (tag, patch, row), deg in row_deg.most_common(10):
            print(f"  {tag} {patch} row={row}: degree={deg}")

    # 7) Probability sanity for measurement-like vs data-like terms
    print("\n[DEM-REPORT] Probability sanity:")
    meas_ones = []
    for p_val, d0 in one_det:
        tag0, _patch0, _row0 = _info(d0)
        if tag0 in ("z_temporal", "x_temporal") and isinstance(p_val, float):
            meas_ones.append(p_val)

    if meas_ones:
        print(
            "  temporal single-detector (measurement-like) prob range: "
            f"{min(meas_ones):.3g} .. {max(meas_ones):.3g}"
        )
        print(f"    sample: {meas_ones[:5]}")
    else:
        print("  no temporal single-detector terms found (or non-numeric probabilities).")

    data_ps = [p for (_tag, _patch, _rpair, p, _d0, _d1) in candidates if isinstance(p, float)]
    if data_ps:
        print(
            "  cross-row (data-like) prob range: "
            f"{min(data_ps):.3g} .. {max(data_ps):.3g}"
        )
    else:
        print("  no numeric probabilities found for cross-row data-like terms.")

    print("\n[DEM-REPORT] ===== End of DEM Health Check (pre-decoder) =====\n")


def _simulate_simple_memory_run(
    qc,
    *,
    distance: int,
    px: float,
    pz: float,
    shots: int,
    seed: int,
    bracket_basis: str,
    demo_basis,
    init_label: str,
    code_type: str,
    p_meas: float,
    use_stim_memory: bool,
):
    """Build and simulate a single-patch memory experiment for summary tables."""
    rounds = distance
    if use_stim_memory:
        circuit, observable_pairs, metadata = build_stim_memory_circuit(
            distance=distance,
            rounds=rounds,
            px=px,
            pz=pz,
            p_meas=p_meas,
            init_label=init_label,
        )
        bracket_map = {"q0": bracket_basis}
    else:
        model = build_surface_code_model(distance, code_type=code_type)
        layout, ops, bracket_map = build_memory_layout(
            model,
            distance=distance,
            rounds=rounds,
            bracket_basis=bracket_basis,
            family=None,
        )
        cfg = PhenomenologicalStimConfig(
            rounds=rounds,
            p_x_error=float(px),
            p_z_error=float(pz),
            p_meas=float(p_meas),
            init_label=init_label,
            bracket_basis=bracket_basis,
            demo_basis=demo_basis,
        )
        gb = GlobalStimBuilder(layout)
        circuit, observable_pairs, metadata = gb.build(
            ops,
            cfg,
            bracket_map,
            qc,
        )
    dem = circuit_to_graphlike_dem(circuit)
    dem = add_spatial_correlations_to_dem(dem, metadata)
    dem = add_boundary_hooks_to_dem(dem, metadata)
    boundary_meta = (metadata.get("boundary_anchors", {}) or {}).get("detector_ids")
    enforce_component_boundaries(dem, explicit_anchor_ids=boundary_meta)
    results = run_logical_simulation(
        circuit=circuit,
        dem=dem,
        metadata=metadata,
        observable_pairs=observable_pairs,
        bracket_map=bracket_map,
        qc=qc,
        shots=int(shots),
        seed=int(seed),
        demo_basis=demo_basis,
        bracket_basis=bracket_basis,
        corr_pairs=None,
        verbose=False,
    )
    return results


def _run_scaling_summary(
    args,
    qc,
    *,
    bracket_basis: str,
    demo_basis,
    init_label: str,
    code_type: str,
) -> None:
    """Run a quick distance-scaling sweep for simple_1q_xzh in verbose mode."""
    if args.benchmark != "simple_1q_xzh":
        return
    distances = [3, 5, 7]
    p_values = [args.px]
    summary_shots = 50000
    print("\n[THRESHOLD-SUMMARY] simple_1q_xzh scaling sanity (shots="
          f"{summary_shots}):")
    header = "      d=" + "   ".join(f"{d:>4}" for d in distances)
    print(header)
    seed_base = int(args.seed) + 1000
    for p_idx, prob in enumerate(p_values):
        row_vals = []
        for d_idx, dist in enumerate(distances):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    res = _simulate_simple_memory_run(
                        qc,
                        distance=dist,
                        px=prob,
                        pz=prob,
                        shots=summary_shots,
                        seed=seed_base + p_idx * 17 + d_idx,
                        bracket_basis=bracket_basis,
                        demo_basis=demo_basis,
                        init_label=init_label,
                        code_type=code_type,
                        p_meas=float(args.p_meas),
                        use_stim_memory=bool(args.stim_memory),
                    )
                ler = res.per_qubit_ler[0] if res.per_qubit_ler else float("nan")
            except Exception as exc:
                print(f"        (d={dist}, p={prob:.2e}) failed: {exc}")
                ler = float("nan")
            row_vals.append(ler)
        row_str = "  ".join(f"{val:8.3e}" if math.isfinite(val) else "   n/a  " for val in row_vals)
        print(f"  p={prob:6.2e}: {row_str}")


def main() -> None:
    args = parse_args()

    run_logical = bool(args.logical_encoding)
    run_transpile = bool(args.transpilation)

    if not run_logical and not run_transpile:
        print("No action requested: enable --logical-encoding true and/or --transpilation true.")
        return

    json_payloads: List[Dict[str, object]] = []

    if run_logical:
        benchmark = instantiate_benchmark(args.benchmark)
        qc = benchmark.get_circuit()

        # Multi-qubit unified path: gate summary and start/end basis
        init_label = (args.init or "0").strip()
        start_basis, _ = parse_init_label(init_label)

        # Global end-basis heuristic for demo auto: toggle basis if any H is present
        any_h = any(ci.operation.name.lower() == "h" for ci in qc.data)
        end_basis = ("X" if start_basis == "Z" else "Z") if any_h else start_basis

        # Unified surgery-based path (works for any number of qubits)
        model = build_surface_code_model(args.distance, code_type=args.code_type)

        plot_heavy_hex_code(model, args.distance, code_type=args.code_type)
        if args.verbose:
            diagnostic_print(model, args)
    

        d = int(args.distance)
        stim_rounds = int(args.rounds) if args.rounds is not None else d
        warmup_rounds = int(args.warmup_rounds)

        # Resolve bracket basis for all patches
        if (args.bracket_basis or "auto").lower() == "auto":
            bracket_basis = start_basis
        else:
            bracket_basis = args.bracket_basis.strip().upper()
        db_mode = (args.demo_basis or "auto").lower()
        if db_mode == "none":
            demo_basis = None
        elif db_mode == "auto":
            demo_basis = end_basis if end_basis != bracket_basis else None
        else:
            # Handle single basis (Z or X) or comma-separated formats like "Z,X"
            demo_str = args.demo_basis.strip().upper()
            demo_basis = demo_str

        # Build a template PatchObject to infer geometry/spacing
        template_patch: PatchObject = PatchObject.from_code_model(model)

        # Compute horizontal spacing based on the actual patch width.
        # Using a fixed offset (e.g., 3.0) can cause overlap at larger d.
        if template_patch.coords:
            xs = [x for (x, _y) in template_patch.coords.values()]
            patch_width = (max(xs) - min(xs)) if xs else 0.0
        else:
            patch_width = 0.0

        # Add a small margin so patches don't touch; reuse ancilla_buffer as a sensible margin
        h_margin = float(args.ancilla_buffer) if hasattr(args, "ancilla_buffer") else 1.0
        h_spacing = patch_width + max(0.5, h_margin)

        # Create patches with geometry-aware horizontal offsets to prevent overlap
        patches = {}
        for i in range(qc.num_qubits):
            # Fresh patch derived from the same template geometry
            patch = template_patch.with_offset(0, i * h_spacing, 0.0)
            patches[f"q{i}"] = patch

        # Start with empty seams - let _auto_generate_seams() handle everything
        seams = {}

        # Optional seam JSON override
        if args.seam_json is not None:
            try:
                seam_cfg = json.loads(args.seam_json.read_text())
                # Expect keys like "rough:q0:q1" mapping to list of [i,j]
                for key, pairs in seam_cfg.items():
                    try:
                        kind, a, b = key.split(":", 2)
                    except Exception:
                        continue
                    seams[(kind, a, b)] = [tuple(map(int, p)) for p in pairs]
            except Exception as _exc:
                pass

        use_memory_layout = (qc.num_qubits == 1)
        use_stim_memory = bool(args.stim_memory and use_memory_layout)
        if use_memory_layout:
            layout, ops, bracket_map = build_memory_layout(
                model,
                distance=d,
                rounds=stim_rounds,
                bracket_basis=bracket_basis,
                family=None,
            )
        else:
            bracket_map = {f"q{i}": bracket_basis for i in range(qc.num_qubits)}
            layout, ops = compile_circuit_to_surgery(
                qc, patches, seams, distance=d, bracket_map=bracket_map,
                warmup_rounds=warmup_rounds, ancilla_strategy=args.cnot_ancilla_strategy, ancilla_buffer=args.ancilla_buffer
            )

        layout.plot(
            annotate=False,         # set True to label global qubit indices
            seams=True,             # draw rough/smooth seam edges
            title=f"Layout for {args.benchmark} (d={d})",
            save_path=PROJECT_ROOT / "plots" / "layout" / args.code_type / f"layout_{args.benchmark}_d{d}.png",
        )

        if args.verbose:
            print("ops")
            print(ops)

        stim_cfg = PhenomenologicalStimConfig(
            rounds=stim_rounds,
            p_x_error=float(args.px),
            p_z_error=float(args.pz),
            p_meas=float(args.p_meas),
            init_label=init_label,
            logical_start=None,
            logical_end=None,
            bracket_basis=bracket_basis,
            demo_basis=demo_basis,
        )

        if use_stim_memory:
            circuit, observable_pairs, metadata = build_stim_memory_circuit(
                distance=d,
                rounds=stim_rounds,
                px=float(args.px),
                pz=float(args.pz),
                p_meas=float(args.p_meas),
                init_label=init_label,
            )
            bracket_map = {"q0": bracket_basis}
        else:
            gb = GlobalStimBuilder(layout)
            circuit, observable_pairs, metadata = gb.build(ops, stim_cfg, bracket_map, qc)

        if args.verbose:
            """
            try:
                print("\n[DEBUG] Stim diagram (detslice-with-ops):")
                print(circuit.diagram('detslice-with-ops', tick=range(0, circuit.num_ticks)))
            except Exception:
                pass
            """

        # Sample DEM and decode
        dem = circuit_to_graphlike_dem(circuit)
        dem = add_spatial_correlations_to_dem(dem, metadata)
        dem = add_boundary_hooks_to_dem(dem, metadata)
        boundary_meta = (metadata.get("boundary_anchors", {}) or {}).get("detector_ids")
        enforce_component_boundaries(dem, explicit_anchor_ids=boundary_meta)

        # Detailed DEM health report (only when verbose)
        if args.verbose:
            _print_dem_health_report(
                dem=dem,
                metadata=metadata,
                project_root=PROJECT_ROOT,
                benchmark_name=args.benchmark,
            )
        # Run simulation
        results = run_logical_simulation(
            circuit=circuit,
            dem=dem,
            metadata=metadata,
            observable_pairs=observable_pairs,
            bracket_map=bracket_map,
            qc=qc,
            shots=int(args.shots),
            seed=int(args.seed),
            demo_basis=demo_basis,
            bracket_basis=bracket_basis,
            corr_pairs=args.corr_pairs,
            verbose=args.verbose,
        )

        # DEM diagnostics (post-decoding)
        if args.verbose:
            try:
                dem_text_post = str(results.dem)
                import re
                err_re = re.compile(r"^\s*error\(([^)]+)\)\s+(.*)$")
                D_re = re.compile(r"\bD(\d+)\b")

                dbg = (metadata.get("mwpm_debug", {}) or {})
                det_ctx = dbg.get("detector_context", {}) or {}

                def _info_post(did: int):
                    meta = det_ctx.get(int(did), {}) or {}
                    tag = meta.get("tag")
                    ctx = meta.get("context", {}) or {}
                    patch = ctx.get("patch")
                    row = ctx.get("row")
                    return tag, patch, row

                two_det_post = []
                for line in dem_text_post.splitlines():
                    m = err_re.match(line)
                    if not m:
                        continue
                    p_str, rhs = m.groups()
                    try:
                        p_val = float(p_str)
                    except Exception:
                        p_val = p_str
                    Ds = [int(x) for x in D_re.findall(rhs)]
                    if len(Ds) == 2:
                        two_det_post.append((p_val, Ds[0], Ds[1]))

                # Count post-aug cross-row candidates
                x_post = []
                z_post = []
                for p_val, d0, d1 in two_det_post:
                    tag0, patch0, row0 = _info_post(d0)
                    tag1, patch1, row1 = _info_post(d1)
                    if not (patch0 == patch1 and isinstance(row0, int) and isinstance(row1, int) and row0 != row1):
                        continue
                    if tag0 == tag1 == "x_temporal":
                        x_post.append((p_val, d0, d1, patch0, tuple(sorted((row0, row1)))))
                    if tag0 == tag1 == "z_temporal":
                        z_post.append((p_val, d0, d1, patch0, tuple(sorted((row0, row1)))))

                from collections import Counter
                bx = Counter((patch, rpair) for _p, _d0, _d1, patch, rpair in x_post)
                bz = Counter((patch, rpair) for _p, _d0, _d1, patch, rpair in z_post)
                print("[DEM-REPORT:POST] cross-row 2-detector candidates (augmented):")
                if bx:
                    print("  x_temporal:")
                    for (patch, rpair), cnt in list(bx.most_common(10)):
                        print(f"    {patch} rows={rpair}  count={cnt}")
                else:
                    print("  x_temporal: none")
                if bz:
                    print("  z_temporal:")
                    for (patch, rpair), cnt in list(bz.most_common(10)):
                        print(f"    {patch} rows={rpair}  count={cnt}")
                else:
                    print("  z_temporal: none")
            except Exception as _exc_post:
                print(f"[DEM-REPORT:POST] scan failed: {_exc_post}")

        # Print structured report
        print_header(args, model, results.dem, metadata, {}, results.cnot_metadata, stim_rounds, int(args.shots))
        if args.verbose:
            dbg = metadata.get("mwpm_debug", {}) or {}
            seam_wraps = dbg.get("seam_wrap_counts", {})
            row_wraps = dbg.get("row_wraps", {})
            if seam_wraps:
                print("[DEBUG] seam wraps:")
                for k, v in seam_wraps.items():
                    print(f"  {k} -> {v}")
            if row_wraps:
                print("[DEBUG] Z row wraps:")
                for q, rows in (row_wraps.get("Z", {}) or {}).items():
                    print(f"  {q}: {rows}")
                print("[DEBUG] X row wraps:")
                for q, rows in (row_wraps.get("X", {}) or {}).items():
                    print(f"  {q}: {rows}")
        # ---- Clarifying preamble (requested vs emitted; operator semantics) ----
        _print_demo_preamble(metadata, stim_cfg)
    
        print_per_qubit_results(args, bracket_map, results.corrected_obs, results.obs_u8, results.preds, int(args.shots))
        print_physics_demo(
            results.demo_meta,
            results.demo_z_bits,
            results.demo_x_bits,
            results.correlation_pairs,
            int(args.shots),
            results.pauli_tracker.frame,
            results.joint_demo_bits,
            virtual_gates_per_qubit=results.pauli_tracker.virtual_gates,
        )
        try:
            decoded_order = [name for name in sorted(bracket_map.keys()) if name in bracket_map]
            print_decoded_logical_distribution(
                decoded_order,
                results.corrected_obs,
                int(args.shots),
                pauli_frame=results.pauli_tracker.frame,
                basis_labels=results.basis_labels,
            )
        except Exception as _exc_dist:
            if args.verbose:
                print(f"[DEBUG] decoded logical distribution unavailable: {_exc_dist}")
        decoder_flip_map = {}
        for patch_name, obs_idx in results.patch_to_obs_idx.items():
            if obs_idx < results.preds.shape[1]:
                decoder_flip_map[patch_name] = results.preds[:, obs_idx]

        print_final_state_distribution(
            metadata.get("final_snapshot", {}),
            results.snapshot_bits,
            results.pauli_tracker.frame,
            int(args.shots),
            apply_frame_correction=True,
            decoder_flips=decoder_flip_map,
        )
        try:
            print_schrodinger_snapshot_distribution(
                metadata.get("final_snapshot", {}),
                results.snapshot_bits,
            )
        except Exception as _exc_sz:
            if args.verbose:
                print(f"[DEBUG] Schrödinger Z snapshot derivation failed: {_exc_sz}")
        print_pauli_frame_audit(results.pauli_tracker.virtual_gates, results.pauli_tracker.frame, results.cnot_metadata)
        
        if args.verbose:
            print_debug_details(args, results.basis_labels, results.obs_u8, results.preds, results.corrected_obs, results.expected_flips)

        if args.verbose and args.benchmark == "simple_1q_xzh":
            try:
                _run_scaling_summary(
                    args,
                    qc,
                    bracket_basis=bracket_basis,
                    demo_basis=demo_basis,
                    init_label=init_label,
                    code_type=args.code_type,
                )
            except Exception as _summary_exc:
                print(f"[THRESHOLD-SUMMARY] skipped: {_summary_exc}")

        # Generate and save detailed JSON report
        # Add snapshot metadata to args for JSON generation
        args.snapshot_meta = metadata.get("final_snapshot", {})

        if args.dump_json:
            detailed_json = generate_detailed_json(
                args, model, metadata, results.cnot_metadata,
                bracket_map, results.corrected_obs, results.obs_u8, results.preds,
                results.demo_z_bits, results.demo_x_bits, results.correlations,
                results.pauli_tracker.virtual_gates, results.pauli_tracker.frame, int(args.shots), stim_rounds,
                results.snapshot_bits
            )
            json_filepath = save_detailed_json(detailed_json, args)
            print(f"\nDetailed report saved to: {json_filepath}")


    if run_transpile:
        benchmark = instantiate_benchmark(args.benchmark)
        logical_metrics = benchmark.compute_logical_metrics()

        target = load_target(args.target)
        config = build_config(target, args)
        transpiler = HeavyHexTranspiler(config)

        logical_circuit = benchmark.get_circuit()
        _best, best_metrics, leaderboard = transpiler.run_baseline(logical_circuit)

        leaderboard_payload = format_leaderboard(leaderboard)
        print(f"Benchmark: {args.benchmark}")
        print(f"Target: {args.target}")
        print(f"Logical metrics: {logical_metrics}")
        print(f"Best mapped metrics: {best_metrics}")
        print("Leaderboard (top candidates):")
        for entry in leaderboard_payload:
            metrics = entry["metrics"]
            print(
                "  #{rank}: depth={depth} twoq={twoq} swaps={swaps} duration_ns={dur}".format(
                    rank=entry["rank"],
                    depth=metrics.get("depth"),
                    twoq=metrics.get("twoq"),
                    swaps=metrics.get("swaps"),
                    dur=metrics.get("duration_ns"),
                )
            )

        json_payloads.append(
            {
                "branch": "transpilation",
                "benchmark": args.benchmark,
                "target": args.target,
                "logical_metrics": logical_metrics,
                "best_metrics": best_metrics,
                "leaderboard": leaderboard_payload,
            }
        )

    if args.dump_json and json_payloads:
        payload_out: List[Dict[str, object]] | Dict[str, object]
        if len(json_payloads) == 1:
            payload_out = json_payloads[0]
        else:
            payload_out = json_payloads
        args.dump_json.write_text(json.dumps(payload_out, indent=2, sort_keys=True))
        print(f"\nDumped JSON snapshot to {args.dump_json}")


if __name__ == "__main__":
    main()
