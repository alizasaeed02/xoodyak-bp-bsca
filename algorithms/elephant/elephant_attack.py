"""
BSCA attack on Elephant Dumbo LFSR.

Per Sarry et al. CARDIS 2023, Section 5 "Attack on Elephant".

Setup:
  - 20-byte initial state (x_0..x_19) is the secret = mask^{0,0}_K
  - Run LFSR forward to generate x_20, x_21, ... (more bytes give more constraints)
  - Compute y_j = x_j XOR x_{j+1}  (mask^{0,1})
  - Compute z_j = x_j XOR x_{j+2}  (mask^{0,2})
  
  - Attacker observes noisy HW of each x_j, y_j, z_j byte
  
Tanner graph (Figure 8 of paper):
  - Variable nodes: bytes x_j, y_j, z_j, and intermediate t_{j+20}
  - Factor types:
    * E_y_j: y_j = x_j XOR x_{j+1}
    * E_z_j: z_j = x_j XOR x_{j+2}
    * E_tx_{j+20}: t_{j+20} = ROL3(x_j) XOR (x_{j+3} << 7)
    * E_ux_{j+20}: x_{j+20} = t_{j+20} XOR (x_{j+13} >> 7)
    * HW factors on every leaked byte

Goal: recover all 20 bytes of the initial state. Metric: rank of correct byte
(out of 256) — same as thesis Table 5.2.
"""

import numpy as np
from numpy.random import default_rng

from dumbo_lfsr import lfsr_run, compute_y_z, rol8
from byte_bp_engine import (
    FactorGraph, HW_BYTE,
    make_y_factor, make_z_factor,
    make_intermediate_factor, make_final_factor,
    make_hw_factor,
)


# ─────────── Build attack ───────────

def attack_dumbo(secret_state, sigma, rng,
                 n_extra_bytes=20, n_iters=15, verbose=False):
    """
    Run BP attack on Dumbo LFSR.
    
    secret_state: list of 20 bytes (the secret)
    sigma: noise level
    rng: numpy random generator
    n_extra_bytes: how many additional LFSR bytes to generate beyond x_19
                   (paper uses some range; more = more constraints)
    n_iters: BP iterations
    
    Returns: list of 20 ranks (one per key byte). Mean rank is the metric.
    """
    # Generate full x trace
    x_trace = lfsr_run(secret_state, num_steps=n_extra_bytes)
    n_x = len(x_trace)  # 20 + n_extra_bytes
    
    # Generate y and z
    y_trace, z_trace = compute_y_z(x_trace)
    n_y = len(y_trace)
    n_z = len(z_trace)
    
    # ── Simulate HW leakage ──
    obs_x = np.array([HW_BYTE[v] for v in x_trace]) + rng.normal(0, sigma, n_x)
    obs_y = np.array([HW_BYTE[v] for v in y_trace]) + rng.normal(0, sigma, n_y)
    obs_z = np.array([HW_BYTE[v] for v in z_trace]) + rng.normal(0, sigma, n_z)
    
    # ── Build factor graph ──
    fg = FactorGraph()
    
    # Add x variables
    for j in range(n_x):
        fg.add_variable(f'x{j}')
    # Add y variables
    for j in range(n_y):
        fg.add_variable(f'y{j}')
    # Add z variables
    for j in range(n_z):
        fg.add_variable(f'z{j}')
    
    # HW factors on every leaked byte
    for j in range(n_x):
        fg.add_factor(f'HWx{j}', [f'x{j}'], make_hw_factor(f'x{j}', obs_x[j], sigma))
    for j in range(n_y):
        fg.add_factor(f'HWy{j}', [f'y{j}'], make_hw_factor(f'y{j}', obs_y[j], sigma))
    for j in range(n_z):
        fg.add_factor(f'HWz{j}', [f'z{j}'], make_hw_factor(f'z{j}', obs_z[j], sigma))
    
    # XOR factors: y_j = x_j XOR x_{j+1}
    for j in range(n_y):
        fg.add_factor(f'Ey{j}', [f'x{j}', f'x{j+1}', f'y{j}'],
                      make_y_factor(f'x{j}', f'x{j+1}', f'y{j}'))
    
    # XOR factors: z_j = x_j XOR x_{j+2}
    for j in range(n_z):
        fg.add_factor(f'Ez{j}', [f'x{j}', f'x{j+2}', f'z{j}'],
                      make_z_factor(f'x{j}', f'x{j+2}', f'z{j}'))
    
    # LFSR feedback factors: for each new byte x_{j+20} (j from 0 to n_extra-1)
    # Add intermediate t_{j+20} variable, then factor for t and final factor for x
    for j in range(n_extra_bytes):
        new_idx = j + 20
        t_name = f't{new_idx}'
        fg.add_variable(t_name)
        # Intermediate: t_{j+20} = ROL3(x_j) XOR (x_{j+3} << 7)
        fg.add_factor(f'Etx{new_idx}',
                      [f'x{j}', f'x{j+3}', t_name],
                      make_intermediate_factor(f'x{j}', f'x{j+3}', t_name))
        # Final: x_{j+20} = t_{j+20} XOR (x_{j+13} >> 7)
        fg.add_factor(f'Eux{new_idx}',
                      [t_name, f'x{j+13}', f'x{new_idx}'],
                      make_final_factor(t_name, f'x{j+13}', f'x{new_idx}'))
    
    # ── Run BP ──
    fg.initialize()
    for it in range(n_iters):
        fg.iterate()
        if verbose:
            ranks = [fg.rank(f'x{j}', secret_state[j]) for j in range(20)]
            print(f"  iter {it+1:2d}: mean rank of 20 key bytes = {np.mean(ranks):.2f}")
    
    # ── Compute final ranks for the 20 key bytes ──
    ranks = [fg.rank(f'x{j}', secret_state[j]) for j in range(20)]
    return ranks


if __name__ == "__main__":
    import time
    rng = default_rng(0)
    print("Elephant Dumbo BSCA attack — demo")
    print("=" * 60)
    print(f"σ = 0.1, 5 trials, 20 extra bytes, 15 BP iterations")
    print()
    
    all_ranks = []
    for trial in range(5):
        secret = list(rng.integers(0, 256, size=20).tolist())
        t0 = time.time()
        ranks = attack_dumbo(secret, sigma=0.1, rng=rng,
                              n_extra_bytes=20, n_iters=15,
                              verbose=(trial == 0))
        elapsed = time.time() - t0
        all_ranks.extend(ranks)
        print(f"trial {trial+1}: mean rank = {np.mean(ranks):.2f}, "
              f"max = {max(ranks)}, ({elapsed:.1f}s)")
    
    print()
    arr = np.array(all_ranks)
    print(f"Overall: mean = {arr.mean():.2f}, std = {arr.std():.2f}, "
          f"median = {np.median(arr):.0f}, max = {arr.max()}")
    print(f"Paper Table 5.2 (σ=0.1): mean=3.54, std=5.97, median=3, max=27")