import sys
print("SCRIPT STARTED", flush=True)

import json
import numpy as np
from scalib.attacks import FactorGraph, BPState
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# PARAMETERS
# ==========================================================
N_BYTES     = 20
EXPERIMENTS = 1000
BP_ITER     = 20
SIGMAS      = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 1.0]

# ==========================================================
# HAMMING WEIGHT TABLE
# ==========================================================
HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.float64)

# ==========================================================
# XOODYAK CHI STEP (byte-level)
# t[i] = k[i] ^ ((~k[i+1]) & k[i+2])
# ==========================================================
def chi_byte(k0, k1, k2):
    return int(k0) ^ ((~int(k1) & 0xFF) & int(k2))

# ==========================================================
# LEAKAGE: HW + Gaussian noise
# ==========================================================
def simulate_leakage(byte_val, sigma):
    return HW[int(byte_val)] + np.random.normal(0, sigma)

# ==========================================================
# BUILD LIKELIHOOD over 256 candidates
# p(obs | byte=v) proportional to exp(-0.5*((HW[v]-obs)/sigma)^2)
# ==========================================================
def build_likelihood(obs, sigma):
    probs = np.exp(-0.5 * ((HW - obs) / sigma) ** 2)
    probs += 1e-300
    probs /= probs.sum()
    return probs.astype(np.float64)

# ==========================================================
# RANK: how many candidates score higher than true value
# ==========================================================
def compute_rank(posterior, true_val):
    return int(np.sum(posterior > posterior[int(true_val)]))

# ==========================================================
# FACTOR GRAPH for Xoodyak chi column
# Variables: k0, k1, k2, t0
# Property:  t0 = k0 ^ (~k1 & k2)
# nc=256 (byte level)
# ==========================================================
GRAPH_STR = """
NC 256
VAR MULTI k0
VAR MULTI k1
VAR MULTI k2
VAR MULTI k1_not
VAR MULTI and_out
VAR MULTI t0
PROPERTY k1_not = !k1
PROPERTY and_out = k1_not & k2
PROPERTY t0 = k0 ^ and_out
"""

# Build the graph once (reuse across experiments)
FG = FactorGraph(GRAPH_STR)

# ==========================================================
# SINGLE EXPERIMENT
# ==========================================================
def run_single(key, sigma):
    """Run SASCA on one key, return per-byte posteriors."""
    N = N_BYTES

    # Compute chi outputs
    t = np.array([
        chi_byte(key[i], key[(i+1) % N], key[(i+2) % N])
        for i in range(N)
    ], dtype=np.uint8)

    # Simulate leakage
    lk_obs = np.array([simulate_leakage(key[i], sigma) for i in range(N)])
    lt_obs = np.array([simulate_leakage(t[i],   sigma) for i in range(N)])

    # Build likelihoods shape (N, 256)
    lk = np.array([build_likelihood(lk_obs[i], sigma) for i in range(N)])
    lt = np.array([build_likelihood(lt_obs[i], sigma) for i in range(N)])

    # Run BP for each chi column (i, i+1, i+2)
    # Accumulate posteriors for k0 across all columns it appears in
    posterior_accum = np.ones((N, 256), dtype=np.float64)

    for i in range(N):
        i0 = i
        i1 = (i + 1) % N
        i2 = (i + 2) % N

        try:
            bp = BPState(FG, nexec=1)

            # Set evidence from leakage
            bp.set_evidence("k0", lk[i0].reshape(1, 256))
            bp.set_evidence("k1", lk[i1].reshape(1, 256))
            bp.set_evidence("k2", lk[i2].reshape(1, 256))
            bp.set_evidence("t0", lt[i0].reshape(1, 256))

            # Run loopy BP
            bp.bp_loopy(BP_ITER, initialize_states=True)

            # Get refined posterior for k0
            dist = bp.get_distribution("k0")
            if dist is not None:
                posterior_accum[i0] *= dist[0]

        except Exception as e:
            # fallback to raw likelihood
            posterior_accum[i0] *= lk[i0]

    # Normalize
    for i in range(N):
        s = posterior_accum[i].sum()
        if s > 0:
            posterior_accum[i] /= s

    return posterior_accum

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    import time

    print(f"Running Xoodyak SCALib SASCA", flush=True)
    print(f"N_BYTES={N_BYTES}, EXPERIMENTS={EXPERIMENTS}, BP_ITER={BP_ITER}", flush=True)
    print("=" * 70, flush=True)

    all_ranks   = {}
    all_success = {}
    sigma_time_sec = {}

    for sigma in SIGMAS:
        ranks    = []
        success  = []

        print(f"\nσ = {sigma:.2f} ...", flush=True)
        t0 = time.perf_counter()

        for exp in range(EXPERIMENTS):
            key = np.random.randint(0, 256, N_BYTES, dtype=np.uint8)

            posteriors = run_single(key, sigma)

            # Per-byte ranks
            byte_ranks = [compute_rank(posteriors[i], key[i]) for i in range(N_BYTES)]
            ranks.extend(byte_ranks)

            # Full key recovered = all bytes rank 0
            recovered = all(r == 0 for r in byte_ranks)
            success.append(int(recovered))

            if (exp + 1) % 100 == 0:
                sr = 100.0 * sum(success) / (exp + 1)
                mr = np.mean(ranks)
                print(f"  {exp+1}/{EXPERIMENTS} | success={sr:.1f}% | mean_rank={mr:.2f}", flush=True)

        all_ranks[sigma]   = np.array(ranks)
        all_success[sigma] = np.array(success)
        sigma_time_sec[sigma] = time.perf_counter() - t0
        print(f"  [σ={sigma:.2f}] elapsed: {sigma_time_sec[sigma]:.1f}s "
              f"({sigma_time_sec[sigma]/EXPERIMENTS*1000:.1f} ms/trial)", flush=True)

    # -------------------------------------------------------
    # PRINT OPTION A: Success Rate
    # -------------------------------------------------------
    print("\n")
    print("=" * 55)
    print("OPTION A — Key Recovery Success Rate")
    print("=" * 55)
    print(f"{'σ':>5} | {'Keys Recovered':>14} | {'Success %':>10}")
    print("-" * 40)
    for sigma in SIGMAS:
        s = all_success[sigma]
        print(f"{sigma:>5.2f} | {s.sum():>14} / {EXPERIMENTS} | {100*s.mean():>9.1f}%")

    # -------------------------------------------------------
    # PRINT OPTION B: Per-Byte Rank Statistics
    # -------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("OPTION B — Per-Byte Rank Statistics")
    print("=" * 70)
    print(f"{'σ':>5} | {'Mean':>7} | {'Std':>7} | {'Q1':>4} | {'Med':>4} | {'Q3':>4} | {'Max':>5}")
    print("-" * 60)
    for sigma in SIGMAS:
        r = all_ranks[sigma]
        print(f"{sigma:>5.2f} | {np.mean(r):>7.2f} | {np.std(r):>7.2f} | "
              f"{int(np.percentile(r,25)):>4} | {int(np.median(r)):>4} | "
              f"{int(np.percentile(r,75)):>4} | {int(np.max(r)):>5}")

    # Save
    np.save("scalib_ranks.npy",   all_ranks)
    np.save("scalib_success.npy", all_success)
    with open("scalib_timing.json", "w") as f:
        json.dump({
            "n_bytes": N_BYTES, "experiments": EXPERIMENTS, "bp_iter": BP_ITER,
            "sigma_time_sec": sigma_time_sec,
            "total_time_sec": sum(sigma_time_sec.values()),
        }, f, indent=2)
    print("\nSaved: scalib_ranks.npy, scalib_success.npy, scalib_timing.json", flush=True)