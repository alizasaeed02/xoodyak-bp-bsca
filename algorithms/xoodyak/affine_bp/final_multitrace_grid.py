"""
Final multi-trace grid. Main method: D2_BP (BP supplies hard decisions
and reliability for OSD basis selection; candidate scoring uses the
exact 8-hypothesis joint log-likelihoods, as throughout this audit).
D0 here uses BP marginals through the existing FIXED rows -- this
matches the published pipeline (BP + fixed-row decode), NOT the exact-
marginal D0 used in an earlier (now superseded) run. D2_exact is kept
as a comparison column.

Observation model (explicitly stated, not a shortcut): each of the 6
per-column observations is generated as N INDEPENDENT noisy Hamming-
weight draws of the SAME true bit (chi_bit_graph.leak_ev's internal
loop over n_traces, log-likelihoods summed) -- this is the literal
multi-trace simulation, not the effective-noise sigma/sqrt(N) shortcut
(that identity was only used earlier as a VERIFICATION of this exact
mechanism, confirmed equal to 1e-9).

Threat model: key AND nonce held fixed across the N traces within one
experiment (repeated processing of the same initialization); nonce
drawn independently per experiment. Does not model fresh-nonce
executions.

Stopping rule: the N-grid advances until D2_BP reaches two consecutive
TESTED N values at 100/100, AND continues further (if needed) until D0
ALSO reaches two consecutive 100/100, or N=150 is reached -- whichever is
later. The live counters used to decide WHEN to stop are separate from
what gets REPORTED: after the sweep, both "first tested N at 100/100"
and "first N starting a run of two consecutive tested-100/100 points"
are recomputed once from the complete recorded sequence
(first_single_and_pair_hit), so a later dip can never silently erase an
earlier true answer the way a live running counter could.
"""
import itertools
import time
import json
from multiprocessing import Pool
import numpy as np

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


def osd_decode(hard_bits, reliability, col_log_scores, C):
    order = np.argsort(-reliability, kind="stable")
    basis = select_basis_in_order(L, order)
    if len(basis) != N_KEY:
        return np.zeros(N_KEY, dtype=np.uint8)
    inv_sel = gf2_inverse(L[basis])
    base_rhs = hard_bits[basis] ^ C[basis]
    candidates_rhs = [base_rhs] + [base_rhs.copy() for _ in range(len(basis))]
    for j in range(len(basis)):
        candidates_rhs[j + 1][j] ^= 1
    best_score, best_key = -np.inf, None
    for cand_rhs in candidates_rhs:
        key_cand = ((inv_sel @ cand_rhs) % 2).astype(np.uint8)
        chi_pred = ((L @ key_cand) % 2) ^ C
        score = 0.0
        for ci, (i0, i1, i2) in enumerate(COORDS):
            h = (int(chi_pred[i0]), int(chi_pred[i1]), int(chi_pred[i2]))
            score += col_log_scores[ci, ALL_HYPS.index(h)]
        if score > best_score:
            best_score, best_key = score, key_cand
    return best_key


def run_trial(args):
    sigma, n_traces, seed = args
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

    from scalib.attacks import BPState
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
        log_scores = np.array(log_scores)
        col_log_scores[ci] = log_scores
        post = np.exp(log_scores - log_scores.max()); post /= post.sum()
        marg = np.zeros((3, 2))
        for idx, (h0, h1, h2) in enumerate(ALL_HYPS):
            marg[0, h0] += post[idx]; marg[1, h1] += post[idx]; marg[2, h2] += post[idx]
        for bit_i, gi in enumerate((i0, i1, i2)):
            hard = int(np.argmax(marg[bit_i])); hard_exact[gi] = hard
            p_h = max(marg[bit_i, hard], CLIP); p_o = max(marg[bit_i, 1 - hard], CLIP)
            rel_exact[gi] = abs(np.log(p_h) - np.log(p_o))

        bp = BPState(_FG, 1)
        bp.set_evidence("a0", ev["a0"]); bp.set_evidence("a1", ev["a1"]); bp.set_evidence("a2", ev["a2"])
        bp.set_evidence("b0", ev["b0"]); bp.set_evidence("b1", ev["b1"]); bp.set_evidence("b2", ev["b2"])
        bp.bp_loopy(30, initialize_states=True)
        for bit_i, gi, name in [(0, i0, "a0"), (1, i1, "a1"), (2, i2, "a2")]:
            dist = bp.get_distribution(name)
            hard = int(np.argmax(dist)); hard_bp[gi] = hard
            p_h = max(float(dist[hard]), CLIP); p_o = max(float(dist[1 - hard]), CLIP)
            rel_bp[gi] = abs(np.log(p_h) - np.log(p_o))

    key_D0 = decode_key(hard_bp, L, C, ROWS_fixed, INV_fixed)
    key_D2_BP = osd_decode(hard_bp, rel_bp, col_log_scores, C)
    key_D2_exact = osd_decode(hard_exact, rel_exact, col_log_scores, C)

    return (int(np.sum(key_D0 != key_bits) == 0),
            int(np.sum(key_D2_BP != key_bits) == 0),
            int(np.sum(key_D2_exact != key_bits) == 0))


def run_point(sigma, n_traces, N=100):
    jobs = [(sigma, n_traces, 3000 + int(sigma * 100) * 1000 + n_traces * 1000 + k) for k in range(N)]
    with Pool(4, initializer=_init) as pool:
        rows = list(pool.imap_unordered(run_trial, jobs))
    return sum(r[0] for r in rows), sum(r[1] for r in rows), sum(r[2] for r in rows)


if __name__ == "__main__":
    SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.25]
    N_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150]
    N_PER_POINT = 100

    results = {}
    t_all = time.time()
    for sigma in SIGMAS:
        points = {}
        consec_D2BP = consec_D0 = 0
        first_D2BP = first_D0 = None
        t0 = time.time()
        tested_order = []
        for n_traces in N_GRID:
            s_D0, s_D2BP, s_D2e = run_point(sigma, n_traces, N_PER_POINT)
            points[n_traces] = {"D0_successes": s_D0, "D2_BP_successes": s_D2BP,
                                "D2_exact_successes": s_D2e, "n": N_PER_POINT}
            tested_order.append(n_traces)
            print(f"sigma={sigma} N={n_traces}: D0={s_D0}/100 D2_BP={s_D2BP}/100 "
                  f"D2_exact={s_D2e}/100 ({time.time()-t0:.0f}s elapsed)", flush=True)

            # STOPPING DECISION ONLY (not used for reporting "first" values --
            # those are recomputed cleanly from `points` after the loop, see
            # first_single_and_pair_hit below). This only needs to know
            # whether we are CURRENTLY on a qualifying streak to decide
            # whether to keep testing higher N for D0's sake.
            consec_D2BP = consec_D2BP + 1 if s_D2BP == 100 else 0
            consec_D0 = consec_D0 + 1 if s_D0 == 100 else 0

            if consec_D2BP >= 2 and (consec_D0 >= 2 or n_traces == N_GRID[-1]):
                print(f"  -> stopping sigma={sigma}: D2_BP two-consecutive met "
                      f"AND (D0 two-consecutive met OR N=150 reached)", flush=True)
                break

        def first_single_and_pair_hit(order, succ_by_n):
            """order: N values in the order actually tested. succ_by_n: dict
            N -> successes (int, out of 100). Returns (first_single, first_pair)
            where first_single is the first tested N with 100/100, and
            first_pair is the first N that starts a run of two CONSECUTIVELY
            TESTED N values both at 100/100 (None if neither occurs in the
            tested range). Computed once from the full recorded sequence, so
            a later miss can never silently erase an earlier true answer."""
            first_single = next((n for n in order if succ_by_n[n] == 100), None)
            first_pair = None
            for a, b in zip(order, order[1:]):
                if succ_by_n[a] == 100 and succ_by_n[b] == 100:
                    first_pair = a
                    break
            return first_single, first_pair

        succ_D2BP_by_n = {n: points[n]["D2_BP_successes"] for n in tested_order}
        succ_D0_by_n = {n: points[n]["D0_successes"] for n in tested_order}
        first_single_D2BP, first_pair_D2BP = first_single_and_pair_hit(tested_order, succ_D2BP_by_n)
        first_single_D0, first_pair_D0 = first_single_and_pair_hit(tested_order, succ_D0_by_n)

        results[str(sigma)] = {
            "points": points,
            "first_N_100pct_D2_BP": first_single_D2BP,
            "first_N_100pct_D2_BP_two_consecutive": first_pair_D2BP,
            "first_N_100pct_D0": first_single_D0,
            "first_N_100pct_D0_two_consecutive": first_pair_D0,
            "elapsed_s": time.time() - t0,
        }

    total_elapsed = time.time() - t_all
    with open("final_multitrace_grid_results.json", "w") as f:
        json.dump({"sigmas": SIGMAS, "n_grid": N_GRID, "n_per_point": N_PER_POINT,
                   "total_elapsed_s": total_elapsed,
                   "observation_model": "N independent noisy Hamming-weight draws per trace "
                   "(chi_bit_graph.leak_ev internal loop over n_traces, log-likelihoods summed) "
                   "-- NOT the effective-noise sigma/sqrt(N) shortcut.",
                   "D0_inference_engine": "BP (matches published pipeline), fixed rows",
                   "main_method": "D2_BP (BP hard decisions/reliability, exact-score OSD reprocessing)",
                   "results": results}, f, indent=2, default=str)
    print(f"\nTotal elapsed: {total_elapsed:.0f}s")
    print("saved final_multitrace_grid_results.json")
