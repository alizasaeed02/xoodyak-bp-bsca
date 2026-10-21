"""
linear_theta_CORRECTED_init.py
=====================================================================
AUDIT FIX (Phase 2). Kept separate from linear_theta.py so the original
is preserved and both can be diffed/compared. THIS MODULE IS THE ONE
ACTUALLY USED by every FINAL script in this folder (final_single_trace_
grid.py, final_multitrace_grid.py, step1-5, item2/item3) -- it is fully
wired in, not a standalone proposal.

Bug found in linear_theta.py's key_bits_to_state(): it builds
state = key(plane 0) || 0 || 0, i.e. literally all-zero for planes 1
and 2. The real Xoodyak Cyclist absorb_key() (verified against
xoodyak-master/include/cyclist.hpp, absorb_key()/down()/up(), which
itself cites the official spec PDF line-by-line) constructs, for a
16-byte key and a 16-byte nonce, BEFORE the first permutation:

  state words 0-3   (plane y=0) = key                (matches current code)
  state words 4-7   (plane y=1) = nonce               (current code: 0)
  state word 8, byte0            = 0x10 (len(nonce)=16, 1 byte)  (current code: 0)
  state word 8, byte1            = 0x01 (Cyclist down() pad marker) (current code: 0)
  state word 11, byte3 (MSB)     = 0x02 (AbsorbKey_Color, domain separation) (current code: 0)
  all other bytes                 = 0

This is Algorithm 3 / absorb_key in the Xoodyak spec: msg = key || nonce
|| len(nonce) is absorbed into a zero state (phase already Up, so no
pre-up()), via a single down() call (33 bytes < R_Kin=44, so one call
suffices), then the state sits untouched until the NEXT up() call
(triggered by the following absorb(associated_data) in the AEAD flow)
applies the actual Xoodoo permutation. That next up() is "the first
attacked permutation" this project's chi-only BP attack targets, so
the state computed here -- not key||0||0 -- is its true input.

Nonce handling: the nonce is PUBLIC in Xoodyak's AEAD (sent in the
clear), so it does not need to be "recovered" by the attacker, exactly
like the length/pad/domain-separation bytes. EVERY final script in this
folder passes an explicit, independently random 16-byte nonce per trial
(drawn from its own rng stream, decode uses the matching per-trial
compute_constant(nonce_bytes)) -- this is the actual threat model used
throughout the final results: nonce public and random per trial, key
and nonce both fixed across repeated traces in a multi-trace experiment
(see final_multitrace_grid.py's threat_model field). key_bits_to_state's
`nonce_bytes=None` default (all-zero nonce) is a convenience fallback
used only by a few early/smaller self-tests in this file's own
__main__ block and audit_phase2_validation.py's c0 baseline -- it is
NOT the construction behind any reported single-trace or multi-trace
result.
=====================================================================
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xoodoo_ref as ref

X, Y, Z = 4, 3, 32
N_KEY = X * Z          # 128
N_CHI = Y * X * Z      # 384

RC = ref.ROUND_CONSTANTS[0]


def key_bits_to_state(key_bits, nonce_bytes=None):
    """nonce_bytes: 16-byte public sequence. Every final script passes an
    explicit, independently random nonce per trial; the all-zero default
    (nonce_bytes=None) is only a convenience for this file's own
    self-tests and is not used by any reported result."""
    state = [[0] * 4, [0] * 4, [0] * 4]
    for i, v in enumerate(key_bits):
        x, z = divmod(i, Z)
        state[0][x] |= int(v) << z

    if nonce_bytes is not None:
        nb = bytes(nonce_bytes)
        assert len(nb) == 16
        for x in range(4):
            lane = int.from_bytes(nb[4 * x:4 * x + 4], "little")
            state[1][x] = lane
    # else: nonce assumed zero (state[1] stays [0,0,0,0])

    # Cyclist absorb_key: msg = key || nonce || len(nonce)=0x10, consumed by
    # one down() call (33 bytes <= R_Kin=44). down() XORs the length byte and
    # the 0x01 pad marker into the next lane after key+nonce (word index 8 =
    # plane y=2, lane x=0), and XORs the AbsorbKey_Color=0x02 domain-
    # separation byte into the top byte of the LAST lane (word 11 = plane
    # y=2, lane x=3) -- these three bytes are always present, regardless of
    # key or nonce value; they are public protocol constants, not secrets.
    state[2][0] ^= 0x10          # len(nonce) byte, byte 0 of the lane
    state[2][0] ^= 0x01 << 8     # Cyclist pad10*1 marker, byte 1 of the lane
    state[2][3] ^= 0x02 << 24    # AbsorbKey_Color, byte 3 (MSB) of the lane
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


def build_affine_map(nonce_bytes=None):
    zero_key = np.zeros(N_KEY, dtype=np.uint8)
    c = state_to_chi_input_bits(key_bits_to_state(zero_key, nonce_bytes))

    L = np.zeros((N_CHI, N_KEY), dtype=np.uint8)
    for i in range(N_KEY):
        e = np.zeros(N_KEY, dtype=np.uint8)
        e[i] = 1
        fi = state_to_chi_input_bits(key_bits_to_state(e, nonce_bytes))
        L[:, i] = fi ^ c
    return L, c


def compute_constant(nonce_bytes):
    """c(nonce) for a GIVEN, attacker-known nonce (key=0 baseline). L is
    independent of nonce (nonce bits enter the same affinely-separate way
    key bits do -- see build_nonce_map's docstring and the audit pilot's
    empirical confirmation that L is unchanged by the state-construction
    fix), so only this per-trial constant needs recomputing; L/rows/inv
    are built once and reused."""
    zero_key = np.zeros(N_KEY, dtype=np.uint8)
    return state_to_chi_input_bits(key_bits_to_state(zero_key, nonce_bytes))


def build_nonce_map():
    """L_n (384x128): same differential construction as build_affine_map's
    L, but flipping nonce bits with key held at zero. Returns (L_n, c0)
    where c0 = value at key=0, nonce=0 (= build_affine_map()'s c with
    nonce_bytes=None, since that already means all-zero nonce)."""
    zero_key = np.zeros(N_KEY, dtype=np.uint8)
    c0 = state_to_chi_input_bits(key_bits_to_state(zero_key, bytes(16)))

    L_n = np.zeros((N_CHI, N_KEY), dtype=np.uint8)
    for i in range(N_KEY):
        nb = bytearray(16)
        byte_i, bit_i = divmod(i, 8)
        nb[byte_i] = 1 << bit_i
        fi = state_to_chi_input_bits(key_bits_to_state(zero_key, bytes(nb)))
        L_n[:, i] = fi ^ c0
    return L_n, c0


def gf2_rank(M):
    """Rank of a binary matrix mod 2 via Gaussian elimination (row echelon,
    counting pivots). M is modified on a copy, not in place."""
    A = M.copy().astype(np.uint8)
    n_rows, n_cols = A.shape
    rank = 0
    for col in range(n_cols):
        piv = None
        for row in range(rank, n_rows):
            if A[row, col]:
                piv = row
                break
        if piv is None:
            continue
        A[[rank, piv]] = A[[piv, rank]]
        for row in range(n_rows):
            if row != rank and A[row, col]:
                A[row] ^= A[rank]
        rank += 1
        if rank == n_rows:
            break
    return rank


if __name__ == "__main__":
    import linear_theta as orig

    L_corr, c_corr = build_affine_map()
    L_orig, c_orig = orig.build_affine_map()

    same_L = np.array_equal(L_corr, L_orig)
    diff_c_bits = np.flatnonzero(c_corr != c_orig)
    print(f"L identical to original: {same_L}")
    print(f"c differs at {len(diff_c_bits)} / {N_CHI} bit positions")
    print(f"c_orig nonzero bits: {np.flatnonzero(c_orig).tolist()}")
    print(f"c_corr nonzero bits: {np.flatnonzero(c_corr).tolist()}")

    # self-test: round trip still exact with the corrected map
    rng = np.random.default_rng(0)
    rows, inv = orig.find_invertible_rows(L_corr)
    ok = True
    for _ in range(100):
        key = rng.integers(0, 2, size=N_KEY).astype(np.uint8)
        chi_in = state_to_chi_input_bits(key_bits_to_state(key))
        assert np.array_equal(chi_in, (L_corr @ key) % 2 ^ c_corr)
        key_hat = orig.decode_key(chi_in, L_corr, c_corr, rows, inv)
        if not np.array_equal(key_hat, key):
            ok = False
    print(f"corrected self_test (100 random keys): {'OK' if ok else 'FAIL'}")
