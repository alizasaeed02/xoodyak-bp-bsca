"""
Final experiment — produces THREE tables in the exact format of the paper:

  Table 4: bits recovered for 64-bit K1 (Alzette A4)
  Table 5: bits recovered for 64-bit K2 (Alzette A5)
  Table 6: bits recovered for 128-bit master key K = K1 || K2

Tables print PROGRESSIVELY as each sigma finishes — you don't have to wait
for the whole run to see the table format.

USAGE: python -u run_experiment.py
"""

import json
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from numpy.random import default_rng

from alzette import alzette_full, ALPHA_A4, ALPHA_A5
from hybird_attack import build_alzette_graph
from final_hybrid_attack import iterate_damped


# ────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────

SIGMAS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 1.0]
N_TRIALS = 1000
N_RESTARTS = 4
N_ITERS = 80
DAMPING = 0.4
N_WORKERS = None
OUTPUT_FILE = "results.json"

# ────────────────────────────────────────────────────────────


PAPER_A4 = {
    0.1:  (57.08, 3.02), 0.15: (57.16, 2.66), 0.2:  (57.20, 2.57),
    0.25: (57.15, 2.76), 0.3:  (57.07, 2.74), 0.35: (56.77, 2.94),
    0.4:  (56.63, 2.81), 0.45: (56.12, 3.01), 0.5:  (55.81, 2.81),
    0.6:  (54.83, 2.94), 0.7:  (54.19, 2.68), 1.0:  (52.86, 3.47),
}
PAPER_A5 = {
    0.1:  (56.98, 2.78), 0.15: (57.05, 2.40), 0.2:  (57.05, 2.43),
    0.25: (56.96, 2.55), 0.3:  (56.82, 2.53), 0.35: (56.59, 2.64),
    0.4:  (56.29, 2.67), 0.45: (55.92, 2.75), 0.5:  (55.53, 2.66),
    0.6:  (54.64, 2.70), 0.7:  (53.86, 2.79), 1.0:  (52.78, 3.29),
}
PAPER_MASTER = {
    0.1:  (114.06, 4.07), 0.15: (114.22, 3.59), 0.2:  (114.26, 3.51),
    0.25: (114.11, 3.76), 0.3:  (113.89, 3.67), 0.35: (113.36, 3.94),
    0.4:  (112.92, 3.82), 0.45: (112.05, 4.09), 0.5:  (111.35, 3.85),
    0.6:  (109.48, 3.97), 0.7:  (108.06, 3.77), 1.0:  (105.65, 4.65),
}


# ─────────── Attack ───────────

def attack_one(l0, r0, alpha, sigma, rng,
               n_restarts=4, n_iters=80, damping=0.4):
    vals = alzette_full(l0, r0, alpha)
    obs = {}
    for w in ('lp1','l1','r1','lp2','l2','r2','lp3','l3','r3','lp4','l4','r4'):
        x = vals[w]
        h = np.array([bin((x >> (8*b)) & 0xFF).count('1') for b in range(4)],
                     dtype=float)
        h += rng.normal(0, sigma, 4)
        obs[w] = h
    
    best_total = 0
    best_l0 = best_r0 = 0
    
    for restart in range(n_restarts):
        fg, l0_bits, r0_bits = build_alzette_graph(obs, alpha, sigma)
        fg.initialize()
        if restart > 0:
            local_rng = np.random.default_rng(restart * 7919)
            for fid in fg.v2f:
                for vn in fg.v2f[fid]:
                    card = fg.var_card[vn]
                    noise = local_rng.dirichlet(np.ones(card) * 5)
                    fg.v2f[fid][vn] = 0.8 * fg.v2f[fid][vn] + 0.2 * noise
                    fg.v2f[fid][vn] = fg.v2f[fid][vn] / fg.v2f[fid][vn].sum()
        for it in range(n_iters):
            iterate_damped(fg, damping=damping)
            l0h = sum((1 if fg.belief(l0_bits[j])[1] > 0.5 else 0) << j for j in range(32))
            r0h = sum((1 if fg.belief(r0_bits[j])[1] > 0.5 else 0) << j for j in range(32))
            total = (32 - bin(l0h ^ l0).count('1')) + (32 - bin(r0h ^ r0).count('1'))
            if total > best_total:
                best_total = total
                best_l0 = l0h
                best_r0 = r0h
    return best_total


def _one_trial(args):
    seed, sigma, n_iters, damping, n_restarts = args
    rng = default_rng(seed)
    l0_a4 = int(rng.integers(0, 2**32))
    r0_a4 = int(rng.integers(0, 2**32))
    l0_a5 = int(rng.integers(0, 2**32))
    r0_a5 = int(rng.integers(0, 2**32))
    sa = attack_one(l0_a4, r0_a4, ALPHA_A4, sigma, rng,
                    n_restarts=n_restarts, n_iters=n_iters, damping=damping)
    sb = attack_one(l0_a5, r0_a5, ALPHA_A5, sigma, rng,
                    n_restarts=n_restarts, n_iters=n_iters, damping=damping)
    return (sa, sb)


# ─────────── Table printing (paper-style box) ───────────

def fmt_table_header(title):
    line = "+-------+--------+--------+-----+-----+-----+-----+-----++--------+--------+"
    return [
        line,
        f"| {title:<107}|",
        line,
        "|   σ   |  Mean  |  Std   | Min | Q1  | Med | Q3  | Max ||  Paper |  Paper |",
        "|       |  ours  |  ours  |     |     |     |     |     ||  Mean  |  Std   |",
        line,
    ]


def fmt_table_row(sigma, arr, paper_ref):
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    p_mean, p_std = paper_ref.get(sigma, (None, None))
    p_mean_str = f"{p_mean:6.2f}" if p_mean is not None else "  --  "
    p_std_str  = f"{p_std:6.2f}" if p_std is not None else "  --  "
    return (f"| {sigma:5.2f} | {arr.mean():6.2f} | {arr.std():6.2f} | "
            f"{int(arr.min()):3d} | {int(q1):3d} | {int(med):3d} | "
            f"{int(q3):3d} | {int(arr.max()):3d} || {p_mean_str} | {p_std_str} |")


def fmt_table_footer():
    return "+-------+--------+--------+-----+-----+-----+-----+-----++--------+--------+"


def print_progressive(results_a4, results_a5, results_master):
    """Print all three tables with whatever rows have been computed so far."""
    print("\n" + "=" * 110, flush=True)
    print("CURRENT RESULTS (updated after each σ completes)", flush=True)
    print("=" * 110, flush=True)
    
    for title, results, paper_ref in [
        ("Table 4 -- Bits recovered for 64-bit K1 of Alzette A4", results_a4, PAPER_A4),
        ("Table 5 -- Bits recovered for 64-bit K2 of Alzette A5", results_a5, PAPER_A5),
        ("Table 6 -- Bits recovered for 128-bit master key K", results_master, PAPER_MASTER),
    ]:
        print()
        for line in fmt_table_header(title):
            print(line, flush=True)
        for sigma, arr in results:
            print(fmt_table_row(sigma, arr, paper_ref), flush=True)
        print(fmt_table_footer(), flush=True)


# ─────────── Main ───────────

def main():
    n_workers = N_WORKERS if N_WORKERS is not None else max(1, cpu_count() - 1)
    print(f"Sparkle BSCA experiment — paper-format tables", flush=True)
    print(f"σ ∈ {SIGMAS}", flush=True)
    print(f"N={N_TRIALS} trials, {N_RESTARTS} restarts × {N_ITERS} iters, damping={DAMPING}",
          flush=True)
    print(f"Parallel workers: {n_workers}", flush=True)
    print(f"Tables will print progressively after each σ completes.", flush=True)
    
    results_a4 = []
    results_a5 = []
    results_master = []
    
    for idx, sigma in enumerate(SIGMAS):
        t0 = time.time()
        print(f"\n[{idx+1}/{len(SIGMAS)}] σ={sigma} running {N_TRIALS} trials...",
              flush=True)
        args_list = [(10**8 + 10**6 * int(sigma*1000) + t, sigma,
                      N_ITERS, DAMPING, N_RESTARTS)
                     for t in range(N_TRIALS)]
        if n_workers > 1:
            with Pool(n_workers) as pool:
                trial_results = pool.map(_one_trial, args_list)
        else:
            trial_results = [_one_trial(a) for a in args_list]
        
        a4_arr = np.array([r[0] for r in trial_results], dtype=int)
        a5_arr = np.array([r[1] for r in trial_results], dtype=int)
        master_arr = a4_arr + a5_arr
        
        results_a4.append((sigma, a4_arr))
        results_a5.append((sigma, a5_arr))
        results_master.append((sigma, master_arr))
        
        print(f"[σ={sigma}] done in {time.time()-t0:.0f}s. "
              f"A4={a4_arr.mean():.2f}, A5={a5_arr.mean():.2f}, "
              f"master={master_arr.mean():.2f}", flush=True)
        
        # Print all three tables with what we have so far
        print_progressive(results_a4, results_a5, results_master)
    
    # Save JSON
    output = {'config': {
        'sigmas': SIGMAS, 'n_trials': N_TRIALS,
        'n_restarts': N_RESTARTS, 'n_iters': N_ITERS, 'damping': DAMPING,
    }, 'rows': []}
    for (s, a4), (_, a5), (_, m) in zip(results_a4, results_a5, results_master):
        output['rows'].append({
            'sigma': s,
            'a4_mean': float(a4.mean()), 'a4_std': float(a4.std()),
            'a4_min': int(a4.min()), 'a4_max': int(a4.max()),
            'a4_all': a4.tolist(),
            'a5_mean': float(a5.mean()), 'a5_std': float(a5.std()),
            'a5_min': int(a5.min()), 'a5_max': int(a5.max()),
            'a5_all': a5.tolist(),
            'master_mean': float(m.mean()), 'master_std': float(m.std()),
            'master_min': int(m.min()), 'master_max': int(m.max()),
            'master_all': m.tolist(),
        })
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_FILE}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()