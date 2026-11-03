"""
Unknown-nonce attack, item 3: single-trace grid, sigma in
{0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60}, n=100/sigma,
SAME seed formula as the known-nonce final_single_trace_grid.py
(3000+int(sigma*100)*1000+k), so trials are paired with the existing
known-nonce D2_BP results at the same sigma/seed.

Reports order-1 (reprocessed) results as the headline "unknown nonce"
numbers, side by side with the already-reported known-nonce D2_BP
numbers (read from final_single_trace_grid_results.json, NOT rerun --
that file is untouched). McNemar computed from this run's own per-trial
outcomes paired against a fresh identical-seed run of the known-nonce
D2_BP pipeline (needed since the known-nonce file didn't save per-trial
records either -- same situation as the earlier audit's item5).
"""
import json
import time
from multiprocessing import Pool
import numpy as np
import scipy.stats as st

from unknown_nonce_core import run_trial_unknown_nonce

# --- known-nonce D2_BP, reusing linear_theta_CORRECTED_init + chi_bit_graph,
# identical seed formula, for a genuine per-trial paired McNemar ---
import itertools
from linear_theta import N_KEY, X, Y, Z, chi_input_index, find_invertible_rows, gf2_inverse, decode_key
from linear_theta_CORRECTED_init import build_affine_map, key_bits_to_state, state_to_chi_input_bits, compute_constant
from chi_bit_graph import build_graph, chi3, leak_ev

CLIP = 1e-12
L_known, _ = build_affine_map(nonce_bytes=bytes(16))
ROWS_known, INV_known = find_invertible_rows(L_known)
ALL_HYPS = list(itertools.product([0, 1], repeat=3))
N_CHI = Y * X * Z
N_COL = X * Z


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


def run_trial_known_nonce_D2BP(sigma, seed):
    """Identical protocol to final_single_trace_grid.py's D2_BP path."""
    from scalib.attacks import BPState
    rng_key = np.random.default_rng(seed)
    key_bits = rng_key.integers(0, 2, size=N_KEY).astype(np.uint8)
    rng_nonce = np.random.default_rng(seed + 500_000_000)
    nonce_bytes = bytes(rng_nonce.integers(0, 256, 16, dtype=np.uint8))
    C = compute_constant(nonce_bytes)
    state = key_bits_to_state(key_bits, nonce_bytes)
    chi_in_bits = state_to_chi_input_bits(state)
    rng_noise = np.random.default_rng(seed + 1_000_000_000)

    hard_bp = np.zeros(N_CHI, dtype=np.uint8); rel_bp = np.zeros(N_CHI, dtype=np.float64)
    col_log_scores = np.zeros((N_COL, 8), dtype=np.float64)
    FG = build_graph()
    for ci, (i0, i1, i2) in enumerate(COORDS):
        a0, a1, a2 = int(chi_in_bits[i0]), int(chi_in_bits[i1]), int(chi_in_bits[i2])
        b0, b1, b2 = chi3(a0, a1, a2)
        ev = {"a0": leak_ev(a0, sigma, rng_noise), "a1": leak_ev(a1, sigma, rng_noise),
              "a2": leak_ev(a2, sigma, rng_noise), "b0": leak_ev(b0, sigma, rng_noise),
              "b1": leak_ev(b1, sigma, rng_noise), "b2": leak_ev(b2, sigma, rng_noise)}
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

    order = np.argsort(-rel_bp, kind="stable")
    basis = select_basis_in_order(L_known, order, N_KEY)
    if len(basis) != N_KEY:
        return int(np.sum(np.zeros(N_KEY, dtype=np.uint8) != key_bits) == 0)
    inv_sel = gf2_inverse(L_known[basis])
    base_rhs = hard_bp[basis] ^ C[basis]
    candidates_rhs = [base_rhs] + [base_rhs.copy() for _ in range(len(basis))]
    for j in range(len(basis)):
        candidates_rhs[j + 1][j] ^= 1
    best_score, best_key = -np.inf, None
    for cand_rhs in candidates_rhs:
        key_cand = ((inv_sel @ cand_rhs) % 2).astype(np.uint8)
        chi_pred = ((L_known @ key_cand) % 2) ^ C
        score = 0.0
        for ci, (i0, i1, i2) in enumerate(COORDS):
            h = (int(chi_pred[i0]), int(chi_pred[i1]), int(chi_pred[i2]))
            score += col_log_scores[ci, ALL_HYPS.index(h)]
        if score > best_score:
            best_score, best_key = score, key_cand
    return int(np.array_equal(best_key, key_bits))


def job_unknown(args):
    sigma, seed = args
    r = run_trial_unknown_nonce(sigma, seed)
    return (sigma, seed, int(r["key_success_order1"]), int(r["nonce_correct_order1"]),
            int(r["key_success_order0"]))


def job_known(args):
    sigma, seed = args
    return (sigma, seed, run_trial_known_nonce_D2BP(sigma, seed))


def cp_two_sided(k, n, alpha=0.05):
    lo = st.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = st.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


if __name__ == "__main__":
    SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    N = 100
    results = {}
    t_all = time.time()

    for sigma in SIGMAS:
        seeds = [3000 + int(sigma * 100) * 1000 + k for k in range(N)]
        t0 = time.time()
        with Pool(4) as pool:
            rows_u = list(pool.imap_unordered(job_unknown, [(sigma, s) for s in seeds]))
        t_unknown = time.time() - t0

        t0 = time.time()
        with Pool(4) as pool:
            rows_k = list(pool.imap_unordered(job_known, [(sigma, s) for s in seeds]))
        t_known = time.time() - t0

        succ_u = {s: ks1 for (_, s, ks1, nc1, ks0) in rows_u}
        nonce_u = {s: nc1 for (_, s, ks1, nc1, ks0) in rows_u}
        succ_u_order0 = {s: ks0 for (_, s, ks1, nc1, ks0) in rows_u}
        succ_k = {s: v for (_, s, v) in rows_k}

        n_unknown_succ = sum(succ_u.values())
        n_unknown_order0_succ = sum(succ_u_order0.values())
        n_nonce_succ = sum(nonce_u.values())
        n_known_succ = sum(succ_k.values())

        only_u = sum(1 for s in seeds if succ_u[s] == 1 and succ_k[s] == 0)
        only_k = sum(1 for s in seeds if succ_u[s] == 0 and succ_k[s] == 1)
        both = sum(1 for s in seeds if succ_u[s] == 1 and succ_k[s] == 1)
        neither = sum(1 for s in seeds if succ_u[s] == 0 and succ_k[s] == 0)
        n_disc = only_u + only_k
        p = st.binomtest(min(only_u, only_k), n_disc, 0.5, alternative="two-sided").pvalue if n_disc > 0 else float("nan")

        ci_u = cp_two_sided(n_unknown_succ, N)
        ci_k = cp_two_sided(n_known_succ, N)

        results[str(sigma)] = {
            "n": N,
            "known_nonce_D2_BP_successes": n_known_succ, "known_nonce_cp95_two_sided": ci_k,
            "unknown_nonce_order1_successes": n_unknown_succ, "unknown_nonce_cp95_two_sided": ci_u,
            "unknown_nonce_order0_successes": n_unknown_order0_succ,
            "nonce_recovered_correctly_order1": n_nonce_succ,
            "discordant": {"only_unknown": only_u, "only_known": only_k, "both": both, "neither": neither},
            "mcnemar_exact_two_sided_p": p,
            "runtime_s": {"unknown_nonce": t_unknown, "known_nonce_D2_BP": t_known},
            "per_trial": [{"seed": s, "unknown_order1": succ_u[s], "unknown_order0": succ_u_order0[s],
                          "nonce_order1": nonce_u[s], "known_D2_BP": succ_k[s]} for s in seeds],
        }
        print(f"sigma={sigma}: known={n_known_succ}/100 unknown(order1)={n_unknown_succ}/100 "
              f"(order0={n_unknown_order0_succ}/100) nonce_ok={n_nonce_succ}/100 "
              f"McNemar p={p:.4f}  t_unknown={t_unknown:.0f}s t_known={t_known:.0f}s", flush=True)

    total_elapsed = time.time() - t_all
    with open("unknown_nonce_step3_single_trace_results.json", "w") as f:
        json.dump({"sigmas": SIGMAS, "n_per_sigma": N, "total_elapsed_s": total_elapsed,
                   "note": "unknown_nonce_order1 is the headline 'unknown nonce' result "
                   "(order-1 OSD reprocessing over the 256-bit u=(key,nonce)); order0 is "
                   "pre-reprocessing, included as a diagnostic. known_nonce_D2_BP is a FRESH "
                   "identical-seed rerun of the known-nonce pipeline (not reading "
                   "final_single_trace_grid_results.json), done specifically to recover "
                   "per-trial pairing for McNemar.",
                   "results": results}, f, indent=2, default=str)
    print(f"\nTotal elapsed: {total_elapsed:.0f}s")
    print("saved unknown_nonce_step3_single_trace_results.json")
