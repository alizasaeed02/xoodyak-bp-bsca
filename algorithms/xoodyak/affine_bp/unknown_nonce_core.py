"""
Shared core for the unknown-nonce (fully blind) attack. The attacker
recovers u = (key, nonce) jointly as 256 unknown bits from
c = M @ u XOR c0, where M = [L_k | L_n] (384x256, full rank, public) and
c0 is the FIXED, nonce-independent constant (public protocol bytes
0x10/0x01/0x02 and the round constant only -- no nonce value is ever
read anywhere in this module).

Per-column pipeline (unchanged from the known-nonce D2_BP attack):
  - same leakage model, same 6 observations/column (chi_bit_graph.leak_ev)
  - same per-column loopy BP (SCALib, 30 iterations) for hard decisions
    AND reliability (|LLR|)
  - same exact 8-hypothesis enumeration to build the per-column joint
    log-likelihood table used ONLY for order-1 candidate scoring

Decoder (unknown-nonce OSD, order 0 then order 1):
  1. Sort all 384 chi-input coordinates by BP reliability (descending,
     stable sort -- ties broken by lower coordinate index, same rule as
     the known-nonce decoder).
  2. Greedily select 256 linearly-independent rows of M in that order
     (matroid-greedy over GF(2), same algorithm as before, just over a
     384x256 matrix instead of 384x128).
  3. Solve u_hat = M_sel^-1 @ (hard_bits[sel] XOR c0[sel]) exactly.
  4. Order-1 reprocessing: 257 candidates (u_hat unflipped, plus each of
     the 256 single-bit flips of the selected hard decisions). Each
     candidate is forward-mapped (chi_pred = M @ u_cand XOR c0) and
     scored by the SAME exact joint column log-likelihoods as the
     known-nonce attack. Highest-scoring candidate wins.

The true nonce is used ONLY after the decoder has produced its output,
to count whether the nonce happened to be recovered correctly -- never
inside selection, solving, or scoring.
"""
import itertools
import numpy as np

from linear_theta import N_KEY, X, Y, Z, chi_input_index, gf2_inverse
from linear_theta_CORRECTED_init import (
    build_affine_map, build_nonce_map, key_bits_to_state, state_to_chi_input_bits,
)
from chi_bit_graph import build_graph, chi3, leak_ev

CLIP = 1e-12
N_NONCE = 128
N_U = N_KEY + N_NONCE
ALL_HYPS = list(itertools.product([0, 1], repeat=3))
N_CHI = Y * X * Z
N_COL = X * Z


def build_M_c0():
    L_k, c0_a = build_affine_map(nonce_bytes=bytes(16))
    L_n, c0_b = build_nonce_map()
    assert np.array_equal(c0_a, c0_b)
    M = np.concatenate([L_k, L_n], axis=1)
    return M, c0_a


M_GLOBAL, C0_GLOBAL = build_M_c0()


def nonce_bytes_to_bits(nonce_bytes):
    return np.unpackbits(np.frombuffer(bytes(nonce_bytes), dtype=np.uint8), bitorder="little")


def column_coords():
    coords = []
    for x in range(X):
        for z in range(Z):
            coords.append((chi_input_index(0, x, z), chi_input_index(1, x, z), chi_input_index(2, x, z)))
    return coords


COORDS = column_coords()


def select_basis_in_order(Mat, order, n_cols):
    reduced, selected = [], []
    for r in order:
        v = Mat[r].copy()
        for pivot_col, pr in reduced:
            if v[pivot_col]:
                v = v ^ pr
        nz = np.flatnonzero(v)
        if len(nz) == 0:
            continue
        pivot_col = nz[0]
        reduced.append((pivot_col, v))
        selected.append(r)
        if len(selected) == n_cols:
            break
    return selected


def osd_unknown_nonce_decode(hard_bits, reliability, col_log_scores, M=M_GLOBAL, c0=C0_GLOBAL):
    """Returns (u_hat_order0, u_hat_order1, basis_rank_ok)."""
    order = np.argsort(-reliability, kind="stable")
    basis = select_basis_in_order(M, order, N_U)
    if len(basis) != N_U:
        z = np.zeros(N_U, dtype=np.uint8)
        return z, z, False

    inv_sel = gf2_inverse(M[basis])
    base_rhs = hard_bits[basis] ^ c0[basis]
    u0 = ((inv_sel @ base_rhs) % 2).astype(np.uint8)

    candidates_rhs = [base_rhs] + [base_rhs.copy() for _ in range(len(basis))]
    for j in range(len(basis)):
        candidates_rhs[j + 1][j] ^= 1
    best_score, best_u = -np.inf, u0
    for cand_rhs in candidates_rhs:
        u_cand = ((inv_sel @ cand_rhs) % 2).astype(np.uint8)
        chi_pred = ((M @ u_cand) % 2) ^ c0
        score = 0.0
        for ci, (i0, i1, i2) in enumerate(COORDS):
            h = (int(chi_pred[i0]), int(chi_pred[i1]), int(chi_pred[i2]))
            score += col_log_scores[ci, ALL_HYPS.index(h)]
        if score > best_score:
            best_score, best_u = score, u_cand
    return u0, best_u, True


def run_trial_unknown_nonce(sigma, seed, n_traces=1):
    """Full pipeline for one trial: draw key+nonce, simulate leakage
    (n_traces independent draws per observation, same as the known-nonce
    multi-trace mechanism), run BP + exact enumeration per column, decode
    via the unknown-nonce OSD above. Returns a dict of results; the true
    nonce is read ONLY in the two 'nonce_correct_*' fields at the end."""
    from scalib.attacks import BPState

    rng_key = np.random.default_rng(seed)
    key_bits = rng_key.integers(0, 2, size=N_KEY).astype(np.uint8)
    rng_nonce = np.random.default_rng(seed + 500_000_000)
    nonce_bytes = bytes(rng_nonce.integers(0, 256, 16, dtype=np.uint8))
    nonce_bits_true = nonce_bytes_to_bits(nonce_bytes)
    state = key_bits_to_state(key_bits, nonce_bytes)
    chi_in_bits = state_to_chi_input_bits(state)
    rng_noise = np.random.default_rng(seed + 1_000_000_000)

    hard_bp = np.zeros(N_CHI, dtype=np.uint8)
    rel_bp = np.zeros(N_CHI, dtype=np.float64)
    col_log_scores = np.zeros((N_COL, 8), dtype=np.float64)

    FG = build_graph()
    for ci, (i0, i1, i2) in enumerate(COORDS):
        a0, a1, a2 = int(chi_in_bits[i0]), int(chi_in_bits[i1]), int(chi_in_bits[i2])
        b0, b1, b2 = chi3(a0, a1, a2)
        ev = {"a0": leak_ev(a0, sigma, rng_noise, n_traces), "a1": leak_ev(a1, sigma, rng_noise, n_traces),
              "a2": leak_ev(a2, sigma, rng_noise, n_traces), "b0": leak_ev(b0, sigma, rng_noise, n_traces),
              "b1": leak_ev(b1, sigma, rng_noise, n_traces), "b2": leak_ev(b2, sigma, rng_noise, n_traces)}

        log_scores = []
        for (h0, h1, h2) in ALL_HYPS:
            bb0, bb1, bb2 = chi3(h0, h1, h2)
            s = (np.log(np.clip(ev["a0"][h0], CLIP, 1.0)) + np.log(np.clip(ev["a1"][h1], CLIP, 1.0)) +
                 np.log(np.clip(ev["a2"][h2], CLIP, 1.0)) + np.log(np.clip(ev["b0"][bb0], CLIP, 1.0)) +
                 np.log(np.clip(ev["b1"][bb1], CLIP, 1.0)) + np.log(np.clip(ev["b2"][bb2], CLIP, 1.0)))
            log_scores.append(s)
        col_log_scores[ci] = np.array(log_scores)

        bp = BPState(FG, 1)
        bp.set_evidence("a0", ev["a0"]); bp.set_evidence("a1", ev["a1"]); bp.set_evidence("a2", ev["a2"])
        bp.set_evidence("b0", ev["b0"]); bp.set_evidence("b1", ev["b1"]); bp.set_evidence("b2", ev["b2"])
        bp.bp_loopy(30, initialize_states=True)
        for bit_i, gi, name in [(0, i0, "a0"), (1, i1, "a1"), (2, i2, "a2")]:
            dist = bp.get_distribution(name)
            hard = int(np.argmax(dist)); hard_bp[gi] = hard
            p_h = max(float(dist[hard]), CLIP); p_o = max(float(dist[1 - hard]), CLIP)
            rel_bp[gi] = abs(np.log(p_h) - np.log(p_o))

    u0, u1, basis_ok = osd_unknown_nonce_decode(hard_bp, rel_bp, col_log_scores)

    key0, nonce0 = u0[:N_KEY], u0[N_KEY:]
    key1, nonce1 = u1[:N_KEY], u1[N_KEY:]

    return {
        "sigma": sigma, "seed": seed, "n_traces": n_traces, "basis_rank_ok": basis_ok,
        "key_success_order0": bool(np.array_equal(key0, key_bits)),
        "key_success_order1": bool(np.array_equal(key1, key_bits)),
        "nonce_correct_order0": bool(np.array_equal(nonce0, nonce_bits_true)),
        "nonce_correct_order1": bool(np.array_equal(nonce1, nonce_bits_true)),
        "key_bit_errors_order1": int(np.sum(key1 != key_bits)),
    }
