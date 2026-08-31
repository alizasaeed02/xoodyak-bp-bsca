"""
Run the Elephant Dumbo BSCA attack and produce Table 5.2 reproduction.

USAGE:
    python3 run_experiment.py

Configure parameters at the top of the file.

Output:
  - Table printed to terminal (matching format of thesis Table 5.2)
  - results.json with all per-trial ranks
"""

import json
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from numpy.random import default_rng

from elephant_attack import attack_dumbo


# ────────────────────────────────────────────────────────────
# CONFIGURE HERE
# ────────────────────────────────────────────────────────────

# Noise levels (paper tested 11 levels: 0.1 to 1.0)
SIGMAS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 1.0]
# Quick mode: SIGMAS = [0.1, 0.3, 0.5, 1.0]

# Number of trials (paper uses 1000)
N_TRIALS = 1000

# How many extra LFSR bytes to generate (more = more constraints)
N_EXTRA_BYTES = 20

# BP iterations
N_ITERS = 15

# Parallel workers
N_WORKERS = None    # None = use cores - 1

OUTPUT_FILE = "results.json"

# ────────────────────────────────────────────────────────────


# Reference values from paper Table 5.2
PAPER_REF = {
    0.1:  {'mean': 3.54,  'std': 5.97,  'q1': 0, 'median': 3,  'q3': 3,  'max': 27},
    0.15: {'mean': 3.54,  'std': 5.97,  'q1': 0, 'median': 3,  'q3': 3,  'max': 27},
    0.2:  {'mean': 3.67,  'std': 6.11,  'q1': 0, 'median': 3,  'q3': 3,  'max': 31},
    0.25: {'mean': 4.95,  'std': 9.31,  'q1': 0, 'median': 3,  'q3': 3,  'max': 97},
    0.3:  {'mean': 6.00,  'std': 10.76, 'q1': 0, 'median': 3,  'q3': 3,  'max': 97},
    0.35: {'mean': 7.46,  'std': 13.10, 'q1': 0, 'median': 3,  'q3': 8,  'max': 97},
    0.4:  {'mean': 9.59,  'std': 15.72, 'q1': 0, 'median': 3,  'q3': 8,  'max': 97},
    0.5:  {'mean': 15.97, 'std': 23.03, 'q1': 3, 'median': 8,  'q3': 31, 'max': 153},
    0.6:  {'mean': 23.65, 'std': 31.41, 'q1': 3, 'median': 8,  'q3': 31, 'max': 157},
    0.7:  {'mean': 33.73, 'std': 40.11, 'q1': 3, 'median': 27, 'q3': 36, 'max': 213},
    1.0:  {'mean': 70.68, 'std': 61.42, 'q1': 31,'median': 36, 'q3': 92, 'max': 246},
}


def _one_trial(args):
    seed, sigma, n_extra, n_iters = args
    rng = default_rng(seed)
    secret = list(rng.integers(0, 256, size=20).tolist())
    ranks = attack_dumbo(secret, sigma, rng,
                         n_extra_bytes=n_extra, n_iters=n_iters)
    return ranks


def main():
    n_workers = N_WORKERS if N_WORKERS is not None else max(1, cpu_count() - 1)
    print("=" * 110)
    print("Elephant Dumbo BSCA reproduction — Table 5.2 of Sarry thesis")
    print("=" * 110)
    print(f"σ ∈ {SIGMAS}, N={N_TRIALS} trials, BP iters={N_ITERS}")
    print(f"Parallel workers: {n_workers}")
    print()
    print(f"  {'σ':>5} | {'Mean':>6} | {'Std':>6} | {'Q1':>3} | {'Med':>4} | {'Q3':>3} | "
          f"{'Max':>4} || {'Paper Mean':>10} | {'Paper Std':>9} | {'Paper Med':>9} | {'time':>5}")
    print(f"  {'-'*5} | {'-'*6} | {'-'*6} | {'-'*3} | {'-'*4} | {'-'*3} | "
          f"{'-'*4} || {'-'*10} | {'-'*9} | {'-'*9} | {'-'*5}")
    
    output = {'config': {'sigmas': SIGMAS, 'n_trials': N_TRIALS, 'n_iters': N_ITERS,
                          'n_extra_bytes': N_EXTRA_BYTES}}
    output['rows'] = []
    
    for sigma in SIGMAS:
        t0 = time.time()
        args = [(int(10**8 * sigma + t), sigma, N_EXTRA_BYTES, N_ITERS)
                for t in range(N_TRIALS)]
        if n_workers > 1:
            with Pool(n_workers) as pool:
                trial_results = pool.map(_one_trial, args)
        else:
            trial_results = [_one_trial(a) for a in args]
        # Flatten: each trial gives 20 ranks → total 20 * N_TRIALS ranks
        all_ranks = np.array([r for trial in trial_results for r in trial], dtype=int)
        q1, med, q3 = np.percentile(all_ranks, [25, 50, 75])
        elapsed = time.time() - t0
        ref = PAPER_REF.get(sigma, {})
        print(f"  {sigma:5.2f} | {all_ranks.mean():6.2f} | {all_ranks.std():6.2f} | "
              f"{int(q1):3d} | {int(med):4d} | {int(q3):3d} | {all_ranks.max():4d} || "
              f"{ref.get('mean', '?'):>10} | {ref.get('std', '?'):>9} | "
              f"{ref.get('median', '?'):>9} | {elapsed:>4.0f}s")
        output['rows'].append({
            'sigma': sigma,
            'mean': float(all_ranks.mean()),
            'std': float(all_ranks.std()),
            'q1': float(q1), 'median': float(med), 'q3': float(q3),
            'max': int(all_ranks.max()),
            'paper': ref,
            'all': all_ranks.tolist(),
        })
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print()
    print(f"Results saved to {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()