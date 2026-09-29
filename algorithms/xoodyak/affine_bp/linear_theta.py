"""
linear_theta.py
=====================================================================
theta -> rho_west -> iota is an AFFINE map over GF(2) (linear + a
constant from iota's round-constant XOR). When the non-key planes are
zero (Xoodyak's key-load: state = key(plane 0) || 0 || 0), this map
sends the 128 key bits to the 384 chi-input bits deterministically:

    chi_input = L(key) XOR c      (L: 384x128 GF(2) matrix, c: 384-bit
                                    constant = the map's value at key=0)

This is exactly the "everything here is invertible from public
information" argument in the Xoodyak paper's Fig. 2/6 -- so this stage
does NOT belong inside the probabilistic BP graph (putting it there
made loopy BP converge to the wrong fixed point, see xoodoo_bp_graph.py
history). It is computed once, here, by direct linear algebra, and
BP is reserved for the one genuinely nonlinear step: chi.
=====================================================================
"""
import os
import sys
import numpy as np

# vendored locally (xoodoo_ref.py, same folder) so this script runs on any
# machine after a plain `git clone`, with no machine-specific absolute path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xoodoo_ref as ref

X, Y, Z = 4, 3, 32
N_KEY = X * Z          # 128
N_CHI = Y * X * Z      # 384

RC = ref.ROUND_CONSTANTS[0]  # the TRUE round-0 constant (0x58) -- confirmed against an
                              # independent C++ reference implementation (xoodyak-master/
                              # include/xoodoo.hpp): RC[] = {0x58, 0x38, ..., 0x12} in
                              # forward round order. This module models only the FIRST
                              # round of the real Xoodoo[12] permutation (everything a
                              # blind attacker sees before the first chi), so it must use
                              # round index 0's constant, not the last one.


def key_bits_to_state(key_bits):
    state = [[0] * 4, [0] * 4, [0] * 4]
    for i, v in enumerate(key_bits):
        x, z = divmod(i, Z)
        state[0][x] |= int(v) << z
    return state


def state_to_chi_input_bits(state):
    A = ref.theta(state)
    A = ref.rho_west(A)
    A = ref.iota(A, RC)
    bits = np.zeros(N_CHI, dtype=np.uint8)
    idx = 0
    for y in range(Y):
        for x in range(X):
            for z in range(Z):
                bits[idx] = (A[y][x] >> z) & 1
                idx += 1
    return bits


def chi_input_index(y, x, z):
    return (y * X + x) * Z + z


def build_affine_map():
    """Returns (L, c): L is (384,128) uint8, c is (384,) uint8."""
    zero_key = np.zeros(N_KEY, dtype=np.uint8)
    c = state_to_chi_input_bits(key_bits_to_state(zero_key))

    L = np.zeros((N_CHI, N_KEY), dtype=np.uint8)
    for i in range(N_KEY):
        e = np.zeros(N_KEY, dtype=np.uint8)
        e[i] = 1
        fi = state_to_chi_input_bits(key_bits_to_state(e))
        L[:, i] = fi ^ c
    return L, c


def gf2_inverse(M):
    """Invert a square binary matrix mod 2 via Gauss-Jordan. Returns inverse or None."""
    n = M.shape[0]
    A = M.copy().astype(np.uint8)
    I = np.eye(n, dtype=np.uint8)
    for col in range(n):
        piv = None
        for row in range(col, n):
            if A[row, col]:
                piv = row
                break
        if piv is None:
            return None
        A[[col, piv]] = A[[piv, col]]
        I[[col, piv]] = I[[piv, col]]
        for row in range(n):
            if row != col and A[row, col]:
                A[row] ^= A[col]
                I[row] ^= I[col]
    return I


def find_invertible_rows(L, rng=None):
    """Deterministically find 128 rows of L (out of 384) that are linearly
    independent, via row reduction with pivoting over the columns. Random
    subset selection is unreliable here (dependent rows are structured, not
    scattered uniformly), so this always finds a valid basis when rank==128."""
    n_rows, n_cols = L.shape
    A = L.copy()
    row_of_col = {}
    pivot_rows = []
    avail = list(range(n_rows))
    for col in range(n_cols):
        piv = None
        for idx, row in enumerate(avail):
            if A[row, col]:
                piv = idx
                break
        if piv is None:
            continue
        row = avail.pop(piv)
        pivot_rows.append(row)
        for r2 in avail:
            if A[r2, col]:
                A[r2] ^= A[row]
    if len(pivot_rows) != n_cols:
        raise RuntimeError(f"L has rank {len(pivot_rows)} < {n_cols}, not injective")
    rows = np.array(pivot_rows)
    inv = gf2_inverse(L[rows])
    if inv is None:
        raise RuntimeError("submatrix unexpectedly singular")
    return rows, inv


def decode_key(chi_input_bits_hat, L, c, rows, inv):
    """Given recovered (hard-decision) chi-input bits, recover the key."""
    rhs = (chi_input_bits_hat[rows] ^ c[rows]).astype(np.uint8)
    key_hat = (inv @ rhs) % 2
    return key_hat.astype(np.uint8)


def self_test(n=100, seed=0):
    rng = np.random.default_rng(seed)
    L, c = build_affine_map()
    rows, inv = find_invertible_rows(L, rng)
    # sanity: L restricted to rows, times inv, is identity
    assert np.array_equal((inv @ L[rows]) % 2, np.eye(L.shape[1], dtype=np.uint8))
    for _ in range(n):
        key = rng.integers(0, 2, size=N_KEY).astype(np.uint8)
        chi_in = state_to_chi_input_bits(key_bits_to_state(key))
        assert np.array_equal(chi_in, (L @ key) % 2 ^ c), "affine map mismatch"
        key_hat = decode_key(chi_in, L, c, rows, inv)
        assert np.array_equal(key_hat, key), "decode mismatch"
    print(f"self_test OK: affine map + GF(2) decode exact on {n} random keys")
    return L, c, rows, inv


# ---------------------------------------------------------------------------
# Independent cross-validation against xoodyak-master/include/xoodoo.hpp
# ---------------------------------------------------------------------------
# This block re-derives the Xoodoo round function DIRECTLY from the C++
# source of an unrelated, independently published reference implementation
# (https://github.com/itzmeanjan/xoodyak, include/xoodoo.hpp -- an
# independent, previously existing implementation, not written for this
# project), rather than re-using anything from our own `ref` module. It then
# compares its output against `ref.xoodoo()` bit-for-bit across the full
# 12-round permutation, on the all-zero state and on random states. This is
# the check the RC=ROUND_CONSTANTS[0] comment above and the README refer to;
# previously it had only been run interactively and never saved as code --
# this block is that missing, real, executable check.
M32 = 0xFFFFFFFF
RC_CPP = [0x58, 0x38, 0x3c0, 0xd0, 0x120, 0x14, 0x60, 0x2c, 0x380, 0xf0, 0x1a0, 0x12]


def _rotl(v, n):
    n %= 32
    return ((v << n) | (v >> (32 - n))) & M32 if n else v & M32


def _cyclic_shift(plane, t, v):
    """Bit at (x,z) moves to (x+t, z+v), per xoodoo.hpp's own docstring."""
    new = [0, 0, 0, 0]
    for x in range(4):
        new[(x + t) % 4] = _rotl(plane[x], v)
    return new


def _theta_cpp(state):
    t1 = [state[0][i] ^ state[1][i] ^ state[2][i] for i in range(4)]
    p0 = _cyclic_shift(t1, 1, 5)
    p1 = _cyclic_shift(t1, 1, 14)
    e = [p0[i] ^ p1[i] for i in range(4)]
    return [[state[y][i] ^ e[i] for i in range(4)] for y in range(3)]


def _rho_cpp(state, t1, v1, t2, v2):
    return [state[0], _cyclic_shift(state[1], t1, v1), _cyclic_shift(state[2], t2, v2)]


def _iota_cpp(state, r_idx):
    new = [state[0][:], state[1][:], state[2][:]]
    new[0][0] ^= RC_CPP[r_idx]
    return new


def _chi_cpp(state):
    b0 = [(~state[1][i] & M32) & state[2][i] for i in range(4)]
    b1 = [(~state[2][i] & M32) & state[0][i] for i in range(4)]
    b2 = [(~state[0][i] & M32) & state[1][i] for i in range(4)]
    return [[state[0][i] ^ b0[i] for i in range(4)],
            [state[1][i] ^ b1[i] for i in range(4)],
            [state[2][i] ^ b2[i] for i in range(4)]]


def _round_cpp(state, r_idx):
    s = _theta_cpp(state)
    s = _rho_cpp(s, 1, 0, 0, 11)
    s = _iota_cpp(s, r_idx)
    s = _chi_cpp(s)
    s = _rho_cpp(s, 0, 1, 2, 8)
    return s


def _permute_cpp(state):
    for r in range(12):
        state = _round_cpp(state, r)
    return state


def validate_against_cpp_reference(n_random=30, seed=0):
    """Runs the full 12-round permutation through both this project's `ref`
    module and an independently re-derived version of xoodyak-master's C++
    reference, on the all-zero state and n_random random states, and prints
    PASS/FAIL -- does not silently assert."""
    rng = np.random.default_rng(seed)
    zero = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    all_ok = True

    out_cpp = _permute_cpp([row[:] for row in zero])
    out_ref = ref.xoodoo(zero, n_rounds=12)
    ok = out_cpp == out_ref
    all_ok &= ok
    print(f"[all-zero state]      match: {'PASS' if ok else 'FAIL'}")

    n_pass = 0
    for i in range(n_random):
        st = [[int(rng.integers(0, 1 << 32)) for _ in range(4)] for _ in range(3)]
        o1 = _permute_cpp([row[:] for row in st])
        o2 = ref.xoodoo(st, n_rounds=12)
        ok = o1 == o2
        n_pass += ok
        if not ok:
            print(f"  [random state {i}] MISMATCH: cpp={o1}  ref={o2}")
    all_ok &= (n_pass == n_random)
    print(f"[{n_random} random states]  match: {n_pass}/{n_random}  "
          f"{'PASS' if n_pass == n_random else 'FAIL'}")

    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'} -- "
          f"ref.xoodoo() {'matches' if all_ok else 'does NOT match'} the "
          f"independent C++ reference (xoodyak-master/include/xoodoo.hpp) "
          f"on the full 12-round permutation.")
    return all_ok


if __name__ == "__main__":
    self_test()
    print()
    validate_against_cpp_reference()
