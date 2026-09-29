"""
run_trace_sweep.py
=====================================================================
Extends run_full_key_sweep.py with the second axis this project uses
for every other cipher (Ascon/Elephant/Sparkle): success rate vs
NUMBER OF TRACES at fixed noise levels, not just success vs sigma at
a single trace. Multiple independent noisy observations of the same
key (repeated measurement, same secret) are averaged in log-space
(chi_bit_graph.leak_ev), which is the standard way to push a BP/SASCA
attack's noise tolerance higher than the single-trace number.

Same pipeline as run_full_key_sweep.py: exact GF(2) linear algebra for
theta/rho_west/iota, bit-level chi-only BP, decode to 128-bit key,
full-key-recovered success criterion.
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


def attack_one(key_bits, sigma, n_traces, rng):
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
            bp.set_evidence("a0", leak_ev(a0, sigma, rng, n_traces))
            bp.set_evidence("a1", leak_ev(a1, sigma, rng, n_traces))
            bp.set_evidence("a2", leak_ev(a2, sigma, rng, n_traces))
            bp.set_evidence("b0", leak_ev(b0, sigma, rng, n_traces))
            bp.set_evidence("b1", leak_ev(b1, sigma, rng, n_traces))
            bp.set_evidence("b2", leak_ev(b2, sigma, rng, n_traces))
            bp.bp_loopy(30, initialize_states=True)

            recovered_bits[chi_input_index(0, x, z)] = int(np.argmax(bp.get_distribution("a0")))
            recovered_bits[chi_input_index(1, x, z)] = int(np.argmax(bp.get_distribution("a1")))
            recovered_bits[chi_input_index(2, x, z)] = int(np.argmax(bp.get_distribution("a2")))

    key_hat = decode_key(recovered_bits, _L, _C, _ROWS, _INV)
    return key_hat


def run_one_experiment(args):
    sigma, n_traces, seed = args
    rng = np.random.default_rng(seed)
    key_bits = rng.integers(0, 2, size=N_KEY).astype(np.uint8)
    key_hat = attack_one(key_bits, sigma, n_traces, rng)
    n_errors = int(np.sum(key_hat != key_bits))
    return (sigma, n_traces, seed, n_errors == 0, n_errors)


if __name__ == "__main__":
    SIGMAS = [0.5, 0.75, 1.0, 1.5, 2.0]
    TRACES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    N_PER_SETTING = 100  # matches the Keccak reproduction's protocol
    jobs = [(s, t, 4000 + int(s * 100) * 10000 + t * 100 + k)
            for s in SIGMAS for t in TRACES for k in range(N_PER_SETTING)]

    print(f"{len(jobs)} runs, sigma in {SIGMAS}, traces in {TRACES}, {N_PER_SETTING} seeds each")
    t0 = time.time()
    results = {(s, t): [] for s in SIGMAS for t in TRACES}
    done = 0
    with Pool(4, initializer=_init) as pool:
        for sigma, n_traces, seed, ok, n_errors in pool.imap_unordered(run_one_experiment, jobs):
            done += 1
            results[(sigma, n_traces)].append((ok, n_errors))
            if done % 50 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {time.time()-t0:.0f}s elapsed", flush=True)

    print("\nsigma\\traces | " + "".join(f"{t:>7}" for t in TRACES))
    summary = {}
    for s in SIGMAS:
        row = f"{s:>11} | "
        for t in TRACES:
            vals = results[(s, t)]
            rate = sum(1 for ok, _ in vals if ok) / len(vals)
            summary[f"{s}_{t}"] = {"rate": rate, "n": len(vals)}
            row += f"{rate:>7.2f}"
        print(row)

    with open(Path(__file__).parent / "results_trace_sweep.json", "w") as f:
        json.dump({"summary": summary, "sigmas": SIGMAS, "traces": TRACES, "n_per_setting": N_PER_SETTING}, f, indent=2)
    print("saved results_trace_sweep.json")
