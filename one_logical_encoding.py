import numpy as np
import stim 
print(stim.__version__)
import pymatching as pm
from qiskit_qec.codes.codebuilders.heavyhex_code_builder import HeavyHexCodeBuilder
from qiskit_qec.linear.symplectic import normalizer
from qiskit_qec.operators.pauli_list import PauliList
from collections import defaultdict, deque


# ----- Helpers -----

def rank_gf2(A: np.ndarray) -> int:
    A = (A.copy() & 1).astype(np.uint8)
    m, ncols = A.shape
    rnk = 0
    for c in range(ncols):
        piv = None
        for i in range(rnk, m):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            continue
        if piv != rnk:
            A[[rnk, piv]] = A[[piv, rnk]]
        for i in range(m):
            if i != rnk and A[i, c]:
                A[i, :] ^= A[rnk, :]
        rnk += 1
    return rnk

def _nullspace_gf2(A: np.ndarray):
    """Return a list of basis vectors for the nullspace of A over GF(2)."""
    A = (A.copy() & 1).astype(np.uint8)
    m, n = A.shape
    R = A.copy()
    piv_col = [-1] * m
    r = 0
    for c in range(n):
        p = None
        for i in range(r, m):
            if R[i, c]:
                p = i; break
        if p is None:
            continue
        if p != r:
            R[[r, p]] = R[[p, r]]
        piv_col[r] = c
        for i in range(m):
            if i != r and R[i, c]:
                R[i, :] ^= R[r, :]
        r += 1
    free = [j for j in range(n) if j not in piv_col[:r]]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        # backsolve pivot rows
        for i in range(r - 1, -1, -1):
            c = piv_col[i]
            if c == -1:
                continue
            if R[i, f]:
                v[c] ^= 1
        basis.append(v)
    return basis


def symp_to_string(l_vec: np.ndarray) -> str:
    n = l_vec.size // 2
    z = l_vec[:n]; x = l_vec[n:]
    out = []
    for zi, xi in zip(z, x):
        if zi and xi: out.append('Y')
        elif zi:      out.append('Z')
        elif xi:      out.append('X')
        else:         out.append('I')
    return ''.join(out)


def init_intent(label: str):
    """Map label in {"0","1","+","-"} to (basis, desired_sign).
    basis in {"Z","X"}; desired_sign in {+1,-1}.
    """
    label = label.strip()
    if label == '0':
        return 'Z', +1
    if label == '1':
        return 'Z', -1
    if label == '+':
        return 'X', +1
    if label == '-':
        return 'X', -1
    raise ValueError("init label must be one of '0','1','+','-'")


def mpp_from_string(circuit: stim.Circuit, pauli_str: str):
    """Append one MPP measuring the Pauli product encoded by `pauli_str`.
    Returns the absolute measurement record index (int), or None if trivial.
    """
    targets = []
    first = True
    for q, ch in enumerate(pauli_str):
        if ch == 'I':
            continue
        if not first:
            targets.append(stim.target_combiner())  # '*'
        if ch == 'X':
            targets.append(stim.target_x(q))
        elif ch == 'Z':
            targets.append(stim.target_z(q))
        elif ch == 'Y':
            targets.append(stim.target_y(q))
        else:
            continue
        first = False
    if not targets:
        return None
    circuit.append_operation("MPP", targets)
    return circuit.num_measurements - 1

def rec_from_abs(circuit: stim.Circuit, idx: int):
    """Convert an absolute measurement index into a REC lookback target."""
    return stim.target_rec(idx - circuit.num_measurements)


def find_logicals_heavyhex(code, S_mat, d):
    """
    Heavy-Hex logical finder (algebraic, robust):
      - Use the *gauge* generator matrix M to enforce commutation with the full gauge group.
      - Find Z-only logical from nullspace of X_M (centralizer condition), not in span of M or S.
      - Find X-only logical from nullspace of Z_M, with odd overlap with Z_L, not in span of M or S.
      - Greedily reduce weight within the stabilizer coset.
      - Raise if weights are not equal to the expected distance d.
    Returns (Z_L_str, X_L_str, ZL_vec, XL_vec).
    """
    n = code.n
    M = (code.generators.matrix.astype(np.uint8) & 1)
    Z_M = M[:, :n]
    X_M = M[:, n:]

    S_Z = (S_mat[:, :n] & 1).astype(np.uint8)
    S_X = (S_mat[:, n:] & 1).astype(np.uint8)

    def row_in_span_gf2(A, v):
        """Return True iff row vector v is in the GF(2) row span of A."""
        if A.size == 0:
            return False
        rA = rank_gf2(A)
        rAv = rank_gf2(np.vstack([A, v & 1]))
        return rAv == rA

    def minimize_weight_in_coset_z(z):
        """Greedy reduce Hamming weight of Z-only vector z by adding Z-stabilizer rows.
        Keep commuting with gauge (X_M @ z == 0)."""
        improved = True
        z = (z & 1).copy()
        while improved:
            improved = False
            for r in S_Z:
                z2 = z ^ r
                # Must remain in centralizer of gauge
                if (X_M.dot(z2) % 2).any():
                    continue
                if z2.sum() < z.sum():
                    z = z2
                    improved = True
            # one more pass is often enough
        return z

    def minimize_weight_in_coset_x(x):
        """Greedy reduce Hamming weight of X-only vector x by adding X-stabilizer rows.
        Keep commuting with gauge (Z_M @ x == 0)."""
        improved = True
        x = (x & 1).copy()
        while improved:
            improved = False
            for r in S_X:
                x2 = x ^ r
                if (Z_M.dot(x2) % 2).any():
                    continue
                if x2.sum() < x.sum():
                    x = x2
                    improved = True
        return x

    # -------- Find Z_L (Z-only) from nullspace of X_M --------
    z_candidates = _nullspace_gf2(X_M)
    z_candidates = [z for z in z_candidates if z.any()]  # drop zero
    if not z_candidates:
        raise RuntimeError("No nontrivial Z logical candidate in nullspace of X_M.")

    best_z = None
    best_w = 10**9
    for z in z_candidates:
        # centralizer is already satisfied by construction: X_M @ z == 0
        z_min = minimize_weight_in_coset_z(z)
        # Discard if in gauge or stabilizer span (as a full symplectic row [z|0])
        full_z = np.zeros(2*n, dtype=np.uint8); full_z[:n] = z_min
        if row_in_span_gf2(M, full_z):
            continue
        if row_in_span_gf2(S_mat, full_z):
            continue
        w = int(z_min.sum())
        if w < best_w:
            best_w = w
            best_z = z_min
    if best_z is None:
        raise RuntimeError("Failed to find a Z logical outside the gauge/stabilizer span.")

    # -------- Find X_L (X-only) from nullspace of Z_M with odd overlap --------
    x_candidates = _nullspace_gf2(Z_M)
    x_candidates = [x for x in x_candidates if x.any()]
    if not x_candidates:
        raise RuntimeError("No nontrivial X logical candidate in nullspace of Z_M.")

    Z_support = best_z
    best_x = None
    best_wx = 10**9
    for x in x_candidates:
        # must anticommute with chosen Z_L: dot(z, x) = 1 (mod 2)
        if (int(np.dot(Z_support, x)) & 1) == 0:
            continue
        x_min = minimize_weight_in_coset_x(x)
        full_x = np.zeros(2*n, dtype=np.uint8); full_x[n:] = x_min
        if row_in_span_gf2(M, full_x):
            continue
        if row_in_span_gf2(S_mat, full_x):
            continue
        w = int(x_min.sum())
        if w < best_wx:
            best_wx = w
            best_x = x_min

    if best_x is None:
        raise RuntimeError("Failed to find an X logical with odd overlap outside gauge/stabilizer span.")

    # Build full symplectic vectors and strings
    ZL_vec = np.zeros(2*n, dtype=np.uint8); ZL_vec[:n] = best_z
    XL_vec = np.zeros(2*n, dtype=np.uint8); XL_vec[n:] = best_x

    Z_L = symp_to_string(ZL_vec)
    X_L = symp_to_string(XL_vec)

    # Enforce expected weights equal to d (as requested)
    wZ = int(best_z.sum())
    wX = int(best_x.sum())
    if not (wZ == d and wX == d):
        raise RuntimeError(f"Logical weights not equal to expected distance d={d}: wZ={wZ}, wX={wX}")

    return Z_L, X_L, ZL_vec, XL_vec



# NOTE:
# This is an *abstract syndrome-only* model using Stim's MPP. We "magically"
# measure the multi-qubit Pauli stabilizers (no ancillas/flags). In a realistic
# schedule, these checks use ancilla (and flags for some X checks).
def build_stim_heavyhex_syndrome(
    code,
    S_Z_stabs,
    S_X_stabs,
    rounds=5,
    pX=1e-3,
    pZ=1e-3,
    init_label: str = None,
    Z_L: str = None,
    X_L: str = None,
):
    n = code.n
    c = stim.Circuit()

    # Decide logical used for initialization and tracking.
    basis = None
    desired_sign = +1
    if init_label is not None:
        basis, desired_sign = init_intent(init_label)
        if basis == 'Z' and Z_L is None:
            raise ValueError("Z_L is required for Z-basis initialization")
        if basis == 'X' and X_L is None:
            raise ValueError("X_L is required for X-basis initialization")
    L_str = Z_L if basis == 'Z' else (X_L if basis == 'X' else None)

    # (optional) place qubits on a line for visual slices
    for q in range(n):
        c.append_operation("QUBIT_COORDS", [q], [q, 0])

    # Per-type histories for temporal detectors
    sz_prev = None
    sx_prev = None

    # Local helpers
    def _measure_set(pauli_list):
        out = []
        for s in pauli_list:
            idx = mpp_from_string(c, s)
            if idx is not None:
                out.append(idx)
        return out

    def _add_detectors(prev_list, curr_list):
        for curr_idx, prev_idx in zip(curr_list, prev_list):
            c.append_operation("DETECTOR", [rec_from_abs(c, prev_idx), rec_from_abs(c, curr_idx)])

    if basis == 'Z':
        # Initial logical Z_L before any Z-half measurements
        c.append_operation("TICK")
        zL_start = mpp_from_string(c, L_str)

        # Z halves (with temporal detectors)
        for t in range(rounds):
            c.append_operation("TICK")
            if pX:
                c.append_operation("X_ERROR", [*range(n)], pX)
            if pZ:
                c.append_operation("Z_ERROR", [*range(n)], pZ)
            sz_curr = _measure_set(S_Z_stabs)
            if t > 0 and sz_prev is not None:
                _add_detectors(sz_prev, sz_curr)
            sz_prev = sz_curr

        # Final logical Z_L after last Z half
        c.append_operation("TICK")
        zL_end = mpp_from_string(c, L_str)
        c.append_operation(
            "OBSERVABLE_INCLUDE",
            [rec_from_abs(c, zL_start), rec_from_abs(c, zL_end)],
            0,
        )

        # X halves AFTER logical is closed (so they don't anticommute with L)
        for t in range(rounds):
            c.append_operation("TICK")
            if pX:
                c.append_operation("X_ERROR", [*range(n)], pX)
            if pZ:
                c.append_operation("Z_ERROR", [*range(n)], pZ)
            sx_curr = _measure_set(S_X_stabs)
            if t > 0 and sx_prev is not None:
                _add_detectors(sx_prev, sx_curr)
            sx_prev = sx_curr

        # Return circuit and the logical (start, end) measurement indices
        return c, [(zL_start, zL_end)]

    elif basis == 'X':
        # Initial logical X_L before any X-half measurements
        c.append_operation("TICK")
        xL_start = mpp_from_string(c, L_str)

        # X halves (with temporal detectors)
        for t in range(rounds):
            c.append_operation("TICK")
            if pX:
                c.append_operation("X_ERROR", [*range(n)], pX)
            if pZ:
                c.append_operation("Z_ERROR", [*range(n)], pZ)
            sx_curr = _measure_set(S_X_stabs)
            if t > 0 and sx_prev is not None:
                _add_detectors(sx_prev, sx_curr)
            sx_prev = sx_curr

        # Final logical X_L after last X half
        c.append_operation("TICK")
        xL_end = mpp_from_string(c, L_str)
        c.append_operation(
            "OBSERVABLE_INCLUDE",
            [rec_from_abs(c, xL_start), rec_from_abs(c, xL_end)],
            0,
        )

        # Z halves AFTER logical is closed
        for t in range(rounds):
            c.append_operation("TICK")
            if pX:
                c.append_operation("X_ERROR", [*range(n)], pX)
            if pZ:
                c.append_operation("Z_ERROR", [*range(n)], pZ)
            sz_curr = _measure_set(S_Z_stabs)
            if t > 0 and sz_prev is not None:
                _add_detectors(sz_prev, sz_curr)
            sz_prev = sz_curr

        # Return circuit and the logical (start, end) measurement indices
        return c, [(xL_start, xL_end)]

    else:
        # No logical tracking requested: alternate Z and X halves each round
        for t in range(rounds):
            # Z half
            c.append_operation("TICK")
            if pX:
                c.append_operation("X_ERROR", [*range(n)], pX)
            if pZ:
                c.append_operation("Z_ERROR", [*range(n)], pZ)
            sz_curr = _measure_set(S_Z_stabs)
            if t > 0 and sz_prev is not None:
                _add_detectors(sz_prev, sz_curr)
            sz_prev = sz_curr

            # X half
            c.append_operation("TICK")
            if pX:
                c.append_operation("X_ERROR", [*range(n)], pX)
            if pZ:
                c.append_operation("Z_ERROR", [*range(n)], pZ)
            sx_curr = _measure_set(S_X_stabs)
            if t > 0 and sx_prev is not None:
                _add_detectors(sx_prev, sx_curr)
            sx_prev = sx_curr

        # Return circuit and no logical observable indices
        return c, []

def _rank_increases(A: np.ndarray, row: np.ndarray) -> bool:
    """Return True iff appending `row` increases GF(2) rank of A (2n columns)."""
    if A.size == 0:
        return row.any()
    return rank_gf2(A) < rank_gf2(np.vstack([A, row & 1]))

def extract_css_stabilizers(S_mat: np.ndarray):
    """
    Build a CSS generating set from an arbitrary stabilizer basis S_mat (s x 2n).
    Returns two lists of Pauli strings: S_Z_stabs (Z-only), S_X_stabs (X-only).
    """
    S_mat = (S_mat & 1).astype(np.uint8)
    s, two_n = S_mat.shape
    n = two_n // 2
    SZ = S_mat[:, :n]
    SX = S_mat[:, n:]

    # Start with any rows already Z-only or X-only
    chosen_full = np.zeros((0, 2*n), dtype=np.uint8)
    Z_rows = []
    X_rows = []

    for r in S_mat:
        z, x = r[:n], r[n:]
        if not x.any():  # Z-only row
            if _rank_increases(chosen_full, r):
                chosen_full = np.vstack([chosen_full, r])
                Z_rows.append(z.copy())
        elif not z.any():  # X-only row
            if _rank_increases(chosen_full, r):
                chosen_full = np.vstack([chosen_full, r])
                X_rows.append(x.copy())

    # Complete Z-only via combinations: coeffs c in Null(SX^T) => Z = c^T SZ, X = 0
    CZ = _nullspace_gf2(SX.T)  # each c has length s
    for c in CZ:
        if not c.any():
            continue
        z = (c @ SZ) & 1  # 1 x n
        if not z.any():
            continue
        full = np.zeros(2*n, dtype=np.uint8)
        full[:n] = z
        if _rank_increases(chosen_full, full):
            chosen_full = np.vstack([chosen_full, full])
            Z_rows.append(z)

    # Complete X-only via combinations: coeffs c in Null(SZ^T) => X = c^T SX, Z = 0
    CX = _nullspace_gf2(SZ.T)
    for c in CX:
        if not c.any():
            continue
        x = (c @ SX) & 1
        if not x.any():
            continue
        full = np.zeros(2*n, dtype=np.uint8)
        full[n:] = x
        if _rank_increases(chosen_full, full):
            chosen_full = np.vstack([chosen_full, full])
            X_rows.append(x)

    # Convert to strings for Stim MPP
    def zrow_to_string(z):
        return ''.join('Z' if zi else 'I' for zi in z.tolist())
    def xrow_to_string(x):
        return ''.join('X' if xi else 'I' for xi in x.tolist())

    S_Z_stabs = [zrow_to_string(z) for z in Z_rows if np.any(z)]
    S_X_stabs = [xrow_to_string(x) for x in X_rows if np.any(x)]

    return S_Z_stabs, S_X_stabs

# Re-run the checker on these logicals
def check_logicals(ZL_vec, XL_vec, S_mat):
    n = ZL_vec.size // 2
    def symp_commute(a, b):
        # Returns True if commute, False if anticommute
        z1, x1 = a[:n], a[n:]
        z2, x2 = b[:n], b[n:]
        return ((np.dot(z1, x2) + np.dot(x1, z2)) % 2) == 0
    # Check commutation with all stabilizers
    commute_ZL = [symp_commute(ZL_vec, S_mat[i]) for i in range(S_mat.shape[0])]
    commute_XL = [symp_commute(XL_vec, S_mat[i]) for i in range(S_mat.shape[0])]
    # Check anticommutation
    anticommute = not symp_commute(ZL_vec, XL_vec)
    # Check not in stabilizer group (i.e., not in row span of S_mat)
    def in_stabilizer_group(vec):
        from scipy.linalg import lu
        # Solve S_mat x = vec over GF(2)
        from scipy.linalg import lstsq
        # We'll use rank check
        A = S_mat.copy()
        b = vec.copy()
        mat = np.vstack([A, b])
        rankA = rank_gf2(A)
        rankAb = rank_gf2(mat)
        return rankA == rankAb
    in_S_ZL = in_stabilizer_group(ZL_vec)
    in_S_XL = in_stabilizer_group(XL_vec)
    # Weights
    wZL = int(np.sum(ZL_vec[:n]) + np.sum(ZL_vec[n:]))
    wXL = int(np.sum(XL_vec[:n]) + np.sum(XL_vec[n:]))
    # Print report
    print("Logical operator checker:")
    print(f"  Z_L commutes with all stabilizers: {all(commute_ZL)}")
    print(f"  X_L commutes with all stabilizers: {all(commute_XL)}")
    print(f"  Z_L and X_L anticommute: {anticommute}")
    print(f"  Z_L in stabilizer group: {in_S_ZL}")
    print(f"  X_L in stabilizer group: {in_S_XL}")
    print(f"  Z_L weight: {wZL}")
    print(f"  X_L weight: {wXL}")


# =============================================================
#  Heavy-Hex  — Phenomenological, Stabilizer-Only Simulation
# =============================================================
# This script:
#   1. Builds a Heavy-Hex code from Qiskit-QEC.
#   2. Extracts stabilizers from the gauge group (normalizer).
#   3. Simulates ideal stabilizer measurements using Stim MPP.
#   4. Builds a Detector Error Model (DEM) and decodes with PyMatching.
#   5. Reports the logical error rate for a chosen logical state.
# =============================================================

# ---------------------- (1) Build the code ----------------------
d = 5
code = HeavyHexCodeBuilder(d=d).build()

n = code.n
generators = code.generators
print(f"Heavy-hex code with d={d} has {n} physical qubits.")
print(f"Number of gauge generators: {len(generators)}")
print("Gauge Generators:")
for i, gen in enumerate(generators):
    print(f"  GG{i}: {gen}")

# ---------------------- (2) Extract Stabilizers ----------------------
# The generator matrix M has shape (m, 2n) and encodes each Pauli
# as a binary symplectic vector [Z | X].
M = code.generators.matrix.astype(np.uint8)

# The normalizer decomposes the full gauge group:
#   - S_mat: true stabilizers (center of gauge group)
#   - Xh_mat, Zh_mat: hyperbolic gauge pairs
S_mat, Xh_mat, Zh_mat = normalizer(M)
S = PauliList(S_mat)

print("Stabilizers (basis):")
for i, p in enumerate(S):
    print(f"  SG{i}: {p}")

# Split stabilizers by Pauli type (Z-only vs X-only)
"""
S_list = [str(p) for p in S]
pauli_weight = lambda s: sum(ch != 'I' for ch in s)
only_Z = lambda s: all(ch in 'IZ' for ch in s)
only_X = lambda s: all(ch in 'IX' for ch in s)

S_Z_stabs = [s for s in S_list if only_Z(s) and pauli_weight(s) > 0]
S_X_stabs = [s for s in S_list if only_X(s) and pauli_weight(s) > 0]
"""
S_Z_stabs, S_X_stabs = extract_css_stabilizers(S_mat)
print(f"Using CSS-projected stabilizers: {len(S_Z_stabs)} Z, {len(S_X_stabs)} X "
      f"(rank(S)={S_mat.shape[0]}, total={len(S_Z_stabs)+len(S_X_stabs)})")


# ---------------------- (3) Heavy-Hex Logical Operators (geometry + algebraic) ----------------------
# Build Z_L from the *geometry* using the weight-2 ZZ gauge edges (bars), then
# pick X_L algebraically to (a) commute with all Z-stabilizers and (b) anticommute with Z_L.
Z_L, X_L, ZL_vec, XL_vec = find_logicals_heavyhex(code, S_mat, d)
print("Chosen Z_L (geometry):", Z_L)
print("Chosen X_L (algebraic from Z_L):", X_L)

# Sanity check the logicals
check_logicals(ZL_vec, XL_vec, S_mat)

# ---------------------- (4) Sanity Check: Code Parameters ----------------------
# Check the standard relation: k = n - s - r
#   n = total physical qubits
#   s = number of stabilizer generators
#   r = number of gauge qubits (hyperbolic pairs)
s = S_mat.shape[0]
rankM = rank_gf2(M)
r = (rankM - s) // 2
k = n - s - r
print(f"n={n}, s={s}, r={r}  =>  k={k}")

# ---------------------- (5) Simulation: Syndrome + Decoding ----------------------
# Choose how to initialize the logical qubit: one of {'0', '1', '+', '-'}.
INIT = '1'  # Change as desired

# Build Stim circuit (abstract stabilizer measurement, no ancillas/flags)
c, obs_pair = build_stim_heavyhex_syndrome(
    code,
    S_Z_stabs,
    S_X_stabs,
    rounds=d,
    pX=1e-4,
    pZ=1e-4,
    init_label=INIT,
    Z_L=Z_L,
    X_L=X_L,
)

# Construct the Detector Error Model (DEM)
dem = c.detector_error_model()
print("detectors:", dem.num_detectors)

# Create a PyMatching decoder from the DEM
matcher = pm.Matching.from_detector_error_model(dem)

# ---------------------- (6) Sampling + Logical Error Metrics ----------------------
shots = 5000  # Number of Monte Carlo samples

# D: detector outputs (syndrome bits)
sam_det = c.compile_detector_sampler()
D = sam_det.sample(shots)

# If an observable pair (logical start/end) was returned, sample those indices manually
if obs_pair:
    # Each pair is (start_idx, end_idx)
    # Stim records measurements in order, so we just check those REC indices
    start_idx, end_idx = obs_pair[0]
    # We get all measurements to extract those two bits for each shot
    full_meas = c.compile_sampler().sample(shots)
    O = (full_meas[:, start_idx] ^ full_meas[:, end_idx]).reshape(-1, 1)
else:
    # No logical observable defined
    O = np.zeros((shots, 1), dtype=np.uint8)

# Decode syndromes to predict logical flips
pred = matcher.decode_batch(D)

# Logical interpretation:
#   - Z-basis: +1 for '0', -1 for '1'
#   - X-basis: +1 for '+', -1 for '-'
want_minus = 1 if INIT in ('1', '-') else 0

# Logical error rate = fraction of incorrect logical outcomes
logical_errors = (pred[:, 0] ^ O[:, 0] ^ want_minus).mean()

# Extra diagnostics
avg_synd_weight = D.sum(axis=1).mean()  # mean number of triggered detectors
click_rate = (D.sum(axis=1) > 0).mean()  # fraction of shots with any detection event

# ---------------------- (7) Report Results ----------------------
print(f"shots={shots}")
print(f"logical_error_rate = {logical_errors:.3e}")
print(f"avg_syndrome_weight = {avg_synd_weight:.3f}")
print(f"click_rate(any_detector) = {click_rate:.3f}")
print("Decoding complete with logical initialization.")


