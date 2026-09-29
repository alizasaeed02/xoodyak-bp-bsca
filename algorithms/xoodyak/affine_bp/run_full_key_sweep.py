"""
run_full_key_sweep.py
=====================================================================
Full pipeline, real 384-bit Xoodyak state:
  1. theta -> rho_west -> iota: exact GF(2) linear algebra (no BP;
     see linear_theta.py for why putting this in BP was wrong).
  2. chi: bit-level BP/SASCA, per column, matching this project's own
     published Tanner graph (chi_bit_graph.py; byte-level leakage was
     tried and found information-theoretically ambiguous -- see that
     file's docstring).
  3. GF(2) decode of the recovered chi-input bits back to the 128-bit
     key (exact linear solve, linear_theta.py).

Metric: full 128-bit KEY recovery success rate vs sigma, single trace
-- the same metric, same sigma range, same "single observation per
secret" threat model as the Kannwischer et al. Keccak reproduction in
../../keccak_repro/, for direct comparison.
=====================================================================
"""
import json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

from linear_theta import (
    build_affine_map, find_invertible_rows, decode_key,
    key_bits_to_state, state_to_chi_input_bits, chi_input_index,
    N_KEY, X, Y, Z,
)
from chi_bit_graph import build_graph, chi3, leak_ev

_L, _C = build_affine_map()
_ROWS, _INV = find_invertible_rows(_L)

_FG = None


def _init():
    global _FG
    _FG = build_graph()


def attack_one(key_bits, sigma, rng):
    state = key_bits_to_state(key_bits)
    chi_in_bits = state_to_chi_input_bits(state)

    recovered_bits = np.zeros(Y * X * Z, dtype=np.uint8)
    for x in range(X):
        for z in range(Z):
            a0 = int(chi_in_bits[chi_input_index(0, x, z)])
            a1 = int(chi_in_bits[chi_input_index(1, x, z)])
            a2 = int(chi_in_bits[chi_input_index(2, x, z)])
            b0, b1, b2 = chi3(a0, a1, a2)

            from scalib.attacks import BPState
            bp = BPState(_FG, 1)
            bp.set_evidence("a0", leak_ev(a0, sigma, rng))
            bp.set_evidence("a1", leak_ev(a1, sigma, rng))
            bp.set_evidence("a2", leak_ev(a2, sigma, rng))
            bp.set_evidence("b0", leak_ev(b0, sigma, rng))
            bp.set_evidence("b1", leak_ev(b1, sigma, rng))
            bp.set_evidence("b2", leak_ev(b2, sigma, rng))
            bp.bp_loopy(30, initialize_states=True)

            recovered_bits[chi_input_index(0, x, z)] = int(np.argmax(bp.get_distribution("a0")))
            recovered_bits[chi_input_index(1, x, z)] = int(np.argmax(bp.get_distribution("a1")))
            recovered_bits[chi_input_index(2, x, z)] = int(np.argmax(bp.get_distribution("a2")))

    key_hat = decode_key(recovered_bits, _L, _C, _ROWS, _INV)
    return key_hat


def run_one_experiment(args):
    sigma, seed = args
    rng = np.random.default_rng(seed)
    key_bits = rng.integers(0, 2, size=N_KEY).astype(np.uint8)
    key_hat = attack_one(key_bits, sigma, rng)
    n_errors = int(np.sum(key_hat != key_bits))
    return (sigma, seed, n_errors == 0, n_errors)


if __name__ == "__main__":
    SIGMAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0]
    N_PER_SIGMA = 100  # matches the Keccak reproduction's protocol (Kannwischer et al., Sec. 5.3: n=100/point)
    jobs = [(s, 3000 + int(s * 100) * 1000 + k) for s in SIGMAS for k in range(N_PER_SIGMA)]

    print(f"{len(jobs)} runs, sigma in {SIGMAS}, {N_PER_SIGMA} seeds each")
    t0 = time.time()
    results = {s: [] for s in SIGMAS}
    done = 0
    with Pool(4, initializer=_init) as pool:
        for sigma, seed, ok, n_errors in pool.imap_unordered(run_one_experiment, jobs):
            done += 1
            results[sigma].append((ok, n_errors))
            if done % 20 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {time.time()-t0:.0f}s elapsed", flush=True)

    print("\nsigma | success_rate | mean_key_errors(/128)")
    summary = {}
    for s in SIGMAS:
        vals = results[s]
        rate = sum(1 for ok, _ in vals if ok) / len(vals)
        mean_err = sum(e for _, e in vals) / len(vals)
        summary[s] = {"rate": rate, "mean_errors": mean_err, "n": len(vals)}
        print(f"{s:>5} | {rate:>12.3f} | {mean_err:>10.2f}")

    with open(Path(__file__).parent / "results_full_key_sweep.json", "w") as f:
        json.dump({"summary": summary, "sigmas": SIGMAS, "n_per_sigma": N_PER_SIGMA}, f, indent=2)
    print("saved results_full_key_sweep.json")
