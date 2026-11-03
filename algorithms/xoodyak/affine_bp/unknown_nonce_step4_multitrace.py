"""
Unknown-nonce attack, item 4: multi-trace. Same threat model as the
known-nonce multi-trace grid (key AND nonce fixed across the N traces
within one experiment; attacker still never uses the true nonce). Same
N grid and the CORRECTED stopping rule (first of two consecutive tested
N values at 100/100, recomputed cleanly from the full sequence -- same
fix as applied to final_multitrace_grid.py after reviewer feedback),
n=100/point, extended to N=200 where needed.

Reports the unknown-nonce order-1 success rate at each point; the
known-nonce D2_BP numbers for comparison are read directly from the
existing final_multitrace_grid_results.json (NOT rerun -- that file is
untouched).
"""
import json
import time
from multiprocessing import Pool

from unknown_nonce_core import run_trial_unknown_nonce

N_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200]
N_PER_POINT = 100


def job(args):
    sigma, n_traces, seed = args
    r = run_trial_unknown_nonce(sigma, seed, n_traces=n_traces)
    return int(r["key_success_order1"])


def run_point(sigma, n_traces, N=100):
    seeds = [3000 + int(sigma * 100) * 1000 + n_traces * 1000 + k for k in range(N)]
    with Pool(4) as pool:
        succs = list(pool.imap_unordered(job, [(sigma, n_traces, s) for s in seeds]))
    return sum(succs)


def first_single_and_pair_hit(order, succ_by_n):
    first_single = next((n for n in order if succ_by_n[n] == 100), None)
    first_pair = None
    for a, b in zip(order, order[1:]):
        if succ_by_n[a] == 100 and succ_by_n[b] == 100:
            first_pair = a
            break
    return first_single, first_pair


if __name__ == "__main__":
    SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.25]
    results = {}
    t_all = time.time()

    for sigma in SIGMAS:
        points = {}
        t0 = time.time()
        tested_order = []
        consec = 0
        for n_traces in N_GRID:
            s = run_point(sigma, n_traces, N_PER_POINT)
            points[n_traces] = {"successes": s, "n": N_PER_POINT}
            tested_order.append(n_traces)
            print(f"sigma={sigma} N={n_traces}: unknown_nonce_order1={s}/100 "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)
            consec = consec + 1 if s == 100 else 0
            if consec >= 2:
                print(f"  -> stopping sigma={sigma}: two consecutive 100/100 reached", flush=True)
                break

        succ_by_n = {n: points[n]["successes"] for n in tested_order}
        first_single, first_pair = first_single_and_pair_hit(tested_order, succ_by_n)
        results[str(sigma)] = {"points": points, "first_single_100pct": first_single,
                               "first_two_consecutive_100pct": first_pair,
                               "elapsed_s": time.time() - t0}

    total_elapsed = time.time() - t_all

    known_nonce_comparison = json.load(open("final_multitrace_grid_results.json"))["results"]
    known_summary = {str(s): known_nonce_comparison[str(s)]["first_N_100pct_D2_BP_two_consecutive"]
                     for s in SIGMAS}

    with open("unknown_nonce_step4_multitrace_results.json", "w") as f:
        json.dump({"sigmas": SIGMAS, "n_grid": N_GRID, "n_per_point": N_PER_POINT,
                   "total_elapsed_s": total_elapsed,
                   "threat_model": "Same as known-nonce multi-trace: key AND nonce fixed "
                   "across all N traces within one experiment; a new (key,nonce) drawn "
                   "independently per experiment. N genuine independent noisy draws "
                   "(chi_bit_graph.leak_ev's n_traces loop), not the effective-noise shortcut. "
                   "Attacker never uses the true nonce anywhere in selection/solving/scoring.",
                   "known_nonce_D2_BP_first_two_consecutive_for_comparison": known_summary,
                   "results": results}, f, indent=2, default=str)
    print(f"\nTotal elapsed: {total_elapsed:.0f}s")
    print("known-nonce comparison (first two-consecutive):", known_summary)
    print("saved unknown_nonce_step4_multitrace_results.json")
