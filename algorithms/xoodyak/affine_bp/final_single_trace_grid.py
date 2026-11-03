"""
Final single-trace grid for the paper's main table.

D0 (BP, fixed rows): the PUBLISHED pipeline -- chi_bit_graph's loopy BP
    supplies hard bit decisions, decoded through the existing FIXED
    rows (linear_theta.find_invertible_rows's ROWS/INV, chosen by plain
    Gaussian pivoting, not reliability).
D2_BP: BP supplies hard decisions AND reliability (|LLR| from BP's own
    posterior) for OSD basis selection; order-1 reprocessing scores
    candidates using the EXACT 8-hypothesis joint log-likelihoods
    (computed directly from the raw observations, independent of which
    engine made the hard decisions) -- this is the paper's main
    pipeline: BP + the reliability decoder.
D2_exact: same OSD decoder, but exact enumeration supplies both the
    hard decisions/reliability AND the candidate scoring -- kept as a
    comparison column, not the main method.

Addendum diagnostics (evaluation only, not fed back into any decoder):
  - wrong hard decisions among D2_BP's/D2_exact's SELECTED 128-bit
    basis, vs the true chi-input bits.
  - for D2_BP/D2_exact: whether the true key is PRESENT among the 129
    OSD candidates vs whether it was the one SELECTED (highest score),
    reported separately.
  - runtime split: local inference (BP or exact), basis selection, the
    GF(2) solve itself (matrix inversion + each candidate's multiply),
    candidate scoring, and total pipeline time.

sigma grid: {0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,
0.70,0.80,1.00}, n=100/sigma, corrected init, random public nonce per
trial. One-sided 95% Clopper-Pearson upper failure bound reported
whenever successes=100/100.

RUNTIME NOTE (fixed after reviewer feedback): D2_BP's t_total now
includes t_score_table (the 8-hypothesis column log-likelihood table
build), which D2_BP's own candidate scoring needs even though a
different engine (BP) supplied the hard decisions -- an earlier version
of this file omitted that cost from D2_BP's total. The
final_single_trace_grid_results.json already on disk when this fix
landed still has the UNDER-reported D2_BP totals; see
item3_runtime_accounting_results.json (20 trials, sigma 0.3 and 0.5) for
the corrected, fully phase-separated breakdown.
"""
import itertools
import time
import json
from multiprocessing import Pool
import numpy as np
import scipy.stats as st

from linear_theta import N_KEY, X, Y, Z, decode_key, chi_input_index, find_invertible_rows, gf2_inverse
from linear_theta_CORRECTED_init import (
    build_affine_map, key_bits_to_state, state_to_chi_input_bits, compute_constant,
)
from chi_bit_graph import build_graph, chi3, leak_ev

CLIP = 1e-12
L, _ = build_affine_map(nonce_bytes=bytes(16))
ROWS_fixed, INV_fixed = find_invertible_rows(L)
ALL_HYPS = list(itertools.product([0, 1], repeat=3))
N_CHI = Y * X * Z
N_COL = X * Z

_FG = None


def _init():
    global _FG
    _FG = build_graph()


def column_coords():
    coords = []
    for x in range(X):
        for z in range(Z):
            coords.append((chi_input_index(0, x, z), chi_input_index(1, x, z), chi_input_index(2, x, z)))
    return coords


COORDS = column_coords()


def select_basis_in_order(Lmat, order):
    n_cols = Lmat.shape[1]
    reduced, selected = [], []
    for r in order:
        v = Lmat[r].copy()
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


def osd_decode_full(hard_bits, reliability, col_log_scores, C, true_key=None, true_chi=None):
    """Returns dict with D1 key, D2 (best) key, timing breakdown, and the
    addendum diagnostics (wrong-basis-bit count, true-key-present-among-
    candidates) when true_key/true_chi are supplied (evaluation only)."""
    t0 = time.perf_counter()
    order = np.argsort(-reliability, kind="stable")
    basis = select_basis_in_order(L, order)
    t_basis = time.perf_counter() - t0
    out = {"t_basis": t_basis}
    if len(basis) != N_KEY:
        z = np.zeros(N_KEY, dtype=np.uint8)
        out.update(key_D1=z, key_D2=z, t_solve=0.0, t_score=0.0,
                   wrong_basis_bits=None, true_key_present=False, true_key_selected=False)
        return out

    t0 = time.perf_counter()
    inv_sel = gf2_inverse(L[basis])
    base_rhs = hard_bits[basis] ^ C[basis]
    key_D1 = ((inv_sel @ base_rhs) % 2).astype(np.uint8)
    t_solve = time.perf_counter() - t0
    out["key_D1"] = key_D1

    if true_chi is not None:
        out["wrong_basis_bits"] = int(np.sum(hard_bits[basis] != true_chi[basis]))
    else:
        out["wrong_basis_bits"] = None

    t0 = time.perf_counter()
    candidates_rhs = [base_rhs] + [base_rhs.copy() for _ in range(len(basis))]
    for j in range(len(basis)):
        candidates_rhs[j + 1][j] ^= 1
    best_score, best_key = -np.inf, key_D1
    true_present = False
    for cand_rhs in candidates_rhs:
        key_cand = ((inv_sel @ cand_rhs) % 2).astype(np.uint8)
        if true_key is not None and np.array_equal(key_cand, true_key):
            true_present = True
        chi_pred = ((L @ key_cand) % 2) ^ C
        score = 0.0
        for ci, (i0, i1, i2) in enumerate(COORDS):
            h = (int(chi_pred[i0]), int(chi_pred[i1]), int(chi_pred[i2]))
            score += col_log_scores[ci, ALL_HYPS.index(h)]
        if score > best_score:
            best_score, best_key = score, key_cand
    t_score = time.perf_counter() - t0
    out.update(key_D2=best_key, t_solve=t_solve, t_score=t_score,
               true_key_present=true_present,
               true_key_selected=(true_key is not None and np.array_equal(best_key, true_key)))
    return out


def run_trial(args):
    sigma, seed = args
    rng_key = np.random.default_rng(seed)
    key_bits = rng_key.integers(0, 2, size=N_KEY).astype(np.uint8)
    rng_nonce = np.random.default_rng(seed + 500_000_000)
    nonce_bytes = bytes(rng_nonce.integers(0, 256, 16, dtype=np.uint8))
    C = compute_constant(nonce_bytes)
    state = key_bits_to_state(key_bits, nonce_bytes)
    chi_in_bits = state_to_chi_input_bits(state)
    rng_noise = np.random.default_rng(seed + 1_000_000_000)

    hard_exact = np.zeros(N_CHI, dtype=np.uint8); rel_exact = np.zeros(N_CHI, dtype=np.float64)
    hard_bp = np.zeros(N_CHI, dtype=np.uint8); rel_bp = np.zeros(N_CHI, dtype=np.float64)
    col_log_scores = np.zeros((N_COL, 8), dtype=np.float64)

    t_exact_infer = t_bp_infer = 0.0
    from scalib.attacks import BPState
    for ci, (i0, i1, i2) in enumerate(COORDS):
        a0, a1, a2 = int(chi_in_bits[i0]), int(chi_in_bits[i1]), int(chi_in_bits[i2])
        b0, b1, b2 = chi3(a0, a1, a2)
        ev = {"a0": leak_ev(a0, sigma, rng_noise), "a1": leak_ev(a1, sigma, rng_noise),
              "a2": leak_ev(a2, sigma, rng_noise), "b0": leak_ev(b0, sigma, rng_noise),
              "b1": leak_ev(b1, sigma, rng_noise), "b2": leak_ev(b2, sigma, rng_noise)}

        t0 = time.perf_counter()
        log_scores = []
        for (h0, h1, h2) in ALL_HYPS:
            bb0, bb1, bb2 = chi3(h0, h1, h2)
            s = (np.log(np.clip(ev["a0"][h0], CLIP, 1.0)) + np.log(np.clip(ev["a1"][h1], CLIP, 1.0)) +
                 np.log(np.clip(ev["a2"][h2], CLIP, 1.0)) + np.log(np.clip(ev["b0"][bb0], CLIP, 1.0)) +
                 np.log(np.clip(ev["b1"][bb1], CLIP, 1.0)) + np.log(np.clip(ev["b2"][bb2], CLIP, 1.0)))
            log_scores.append(s)
        log_scores = np.array(log_scores)
        col_log_scores[ci] = log_scores
        post = np.exp(log_scores - log_scores.max()); post /= post.sum()
        marg = np.zeros((3, 2))
        for idx, (h0, h1, h2) in enumerate(ALL_HYPS):
            marg[0, h0] += post[idx]; marg[1, h1] += post[idx]; marg[2, h2] += post[idx]
        t_exact_infer += time.perf_counter() - t0
        for bit_i, gi in enumerate((i0, i1, i2)):
            hard = int(np.argmax(marg[bit_i])); hard_exact[gi] = hard
            p_h = max(marg[bit_i, hard], CLIP); p_o = max(marg[bit_i, 1 - hard], CLIP)
            rel_exact[gi] = abs(np.log(p_h) - np.log(p_o))

        t0 = time.perf_counter()
        bp = BPState(_FG, 1)
        bp.set_evidence("a0", ev["a0"]); bp.set_evidence("a1", ev["a1"]); bp.set_evidence("a2", ev["a2"])
        bp.set_evidence("b0", ev["b0"]); bp.set_evidence("b1", ev["b1"]); bp.set_evidence("b2", ev["b2"])
        bp.bp_loopy(30, initialize_states=True)
        t_bp_infer += time.perf_counter() - t0
        for bit_i, gi, name in [(0, i0, "a0"), (1, i1, "a1"), (2, i2, "a2")]:
            dist = bp.get_distribution(name)
            hard = int(np.argmax(dist)); hard_bp[gi] = hard
            p_h = max(float(dist[hard]), CLIP); p_o = max(float(dist[1 - hard]), CLIP)
            rel_bp[gi] = abs(np.log(p_h) - np.log(p_o))

    t0 = time.perf_counter()
    key_D0 = decode_key(hard_bp, L, C, ROWS_fixed, INV_fixed)
    t_D0_solve = time.perf_counter() - t0

    res_exact = osd_decode_full(hard_exact, rel_exact, col_log_scores, C, key_bits, chi_in_bits)
    res_bp = osd_decode_full(hard_bp, rel_bp, col_log_scores, C, key_bits, chi_in_bits)

    def errs(k):
        return int(np.sum(k != key_bits))

    return {
        "sigma": sigma, "seed": seed,
        "D0": {"full_success": errs(key_D0) == 0, "key_bit_errors": errs(key_D0),
              "t_infer": t_bp_infer, "t_solve": t_D0_solve,
              "t_total": t_bp_infer + t_D0_solve},
        "D2_BP": {"full_success": errs(res_bp["key_D2"]) == 0, "key_bit_errors": errs(res_bp["key_D2"]),
                 "wrong_basis_bits": res_bp["wrong_basis_bits"],
                 "true_key_present": res_bp["true_key_present"],
                 "true_key_selected": res_bp["true_key_selected"],
                 "t_infer": t_bp_infer, "t_score_table": t_exact_infer,
                 "t_basis": res_bp["t_basis"], "t_solve": res_bp["t_solve"],
                 "t_score": res_bp["t_score"],
                 # FULLY INCLUSIVE total: BP inference + the 8-hypothesis
                 # score-table build (t_exact_infer -- required for D2_BP's
                 # OWN candidate scoring below, not optional; see item 3 of
                 # the reviewer follow-up / AUDIT_REPORT.md section 6) +
                 # basis selection + solve + candidate scoring.
                 "t_total": t_bp_infer + t_exact_infer + res_bp["t_basis"] + res_bp["t_solve"] + res_bp["t_score"]},
        "D2_exact": {"full_success": errs(res_exact["key_D2"]) == 0, "key_bit_errors": errs(res_exact["key_D2"]),
                    "wrong_basis_bits": res_exact["wrong_basis_bits"],
                    "true_key_present": res_exact["true_key_present"],
                    "true_key_selected": res_exact["true_key_selected"],
                    "t_infer": t_exact_infer, "t_basis": res_exact["t_basis"], "t_solve": res_exact["t_solve"],
                    "t_score": res_exact["t_score"],
                    "t_total": t_exact_infer + res_exact["t_basis"] + res_exact["t_solve"] + res_exact["t_score"]},
    }


def cp_one_sided_upper_failure(successes, n, alpha=0.05):
    failures = n - successes
    return float(st.beta.ppf(1 - alpha, failures + 1, successes)) if failures < n else 1.0


def cp_two_sided(k, n, alpha=0.05):
    lo = st.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = st.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


if __name__ == "__main__":
    SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 1.00]
    N = 100
    results = {}
    t_all = time.time()
    for sigma in SIGMAS:
        jobs = [(sigma, 3000 + int(sigma * 100) * 1000 + k) for k in range(N)]
        t0 = time.time()
        with Pool(4, initializer=_init) as pool:
            trials = list(pool.imap_unordered(run_trial, jobs))
        elapsed = time.time() - t0

        summary = {}
        for m in ["D0", "D2_BP", "D2_exact"]:
            succ = sum(1 for t in trials if t[m]["full_success"])
            cp_upper_fail = cp_one_sided_upper_failure(succ, N) if succ == N else None
            ci2 = cp_two_sided(succ, N)
            mean_err = float(np.mean([t[m]["key_bit_errors"] for t in trials]))
            mean_total_t = float(np.mean([t[m]["t_total"] for t in trials]))
            entry = {"successes": succ, "n": N, "rate": succ / N,
                    "one_sided_95ci_upper_failure_bound": cp_upper_fail,
                    "two_sided_95ci": ci2, "mean_key_bit_errors": mean_err,
                    "mean_total_runtime_s": mean_total_t}
            if m != "D0":
                wbb = [t[m]["wrong_basis_bits"] for t in trials if t[m]["wrong_basis_bits"] is not None]
                present = sum(1 for t in trials if t[m]["true_key_present"])
                selected = sum(1 for t in trials if t[m]["true_key_selected"])
                entry.update(mean_wrong_basis_bits=float(np.mean(wbb)) if wbb else None,
                             true_key_present_count=present, true_key_selected_count=selected,
                             mean_t_infer=float(np.mean([t[m]["t_infer"] for t in trials])),
                             mean_t_basis=float(np.mean([t[m]["t_basis"] for t in trials])),
                             mean_t_solve=float(np.mean([t[m]["t_solve"] for t in trials])),
                             mean_t_score=float(np.mean([t[m]["t_score"] for t in trials])))
            summary[m] = entry
        results[str(sigma)] = {"n": N, "elapsed_s": elapsed, "summary": summary}
        print(f"\nsigma={sigma} (n={N}, {elapsed:.1f}s):")
        for m, v in summary.items():
            cp = v["one_sided_95ci_upper_failure_bound"]
            cp_str = f" 1-sided-CP-upper-fail={cp:.4f}" if cp is not None else ""
            print(f"  {m}: {v['successes']}/{N} ({v['rate']:.2%}){cp_str} mean_err={v['mean_key_bit_errors']:.2f} "
                  f"total_rt={v['mean_total_runtime_s']*1000:.2f}ms")

    total_elapsed = time.time() - t_all
    with open("final_single_trace_grid_results.json", "w") as f:
        json.dump({"sigmas": SIGMAS, "n_per_sigma": N, "total_elapsed_s": total_elapsed,
                   "D0_inference_engine": "BP (chi_bit_graph loopy BP), fixed rows -- matches the published pipeline",
                   "results": results}, f, indent=2, default=str)
    print(f"\nTotal elapsed: {total_elapsed:.0f}s")
    print("saved final_single_trace_grid_results.json")
