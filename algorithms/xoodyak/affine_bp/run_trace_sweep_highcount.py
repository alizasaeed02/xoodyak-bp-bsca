"""
Extends run_trace_sweep.py's grid (traces 1-50) to the higher trace counts
needed to find the exact 100%-recovery crossover for sigma=1.0, 1.5, 2.0, 2.25.
Saved as its own JSON so every number quoted anywhere is backed by a file,
not just a printed console line from an earlier interactive session.
"""
import json, time
from pathlib import Path
from multiprocessing import Pool
from run_trace_sweep import run_one_experiment, _init

if __name__ == "__main__":
    POINTS = [
        (1.0, 25), (1.0, 30),
        (1.5, 60), (1.5, 70), (1.5, 80), (1.5, 100),
        (2.0, 55), (2.0, 60), (2.0, 65), (2.0, 70), (2.0, 75), (2.0, 100), (2.0, 150), (2.0, 200),
        (2.25, 60), (2.25, 70), (2.25, 80), (2.25, 90), (2.25, 100), (2.25, 150), (2.25, 200),
    ]
    N = 50
    jobs = [(s, t, 9700 + int(s * 100) * 10000 + t * 100 + k) for (s, t) in POINTS for k in range(N)]

    print(f"{len(jobs)} runs, {len(POINTS)} (sigma,traces) points, n={N} each")
    t0 = time.time()
    results = {p: [] for p in POINTS}
    done = 0
    with Pool(4, initializer=_init) as pool:
        for sigma, n_traces, seed, ok, n_errors in pool.imap_unordered(run_one_experiment, jobs):
            done += 1
            results[(sigma, n_traces)].append(ok)
            if done % 25 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {time.time()-t0:.0f}s elapsed", flush=True)

    summary = {}
    print("\nsigma  traces  rate")
    for (s, t) in POINTS:
        vals = results[(s, t)]
        rate = sum(vals) / len(vals)
        summary[f"{s}_{t}"] = {"rate": rate, "n": len(vals)}
        print(f"{s:>5}  {t:>6}  {rate:.2f}")

    with open(Path(__file__).parent / "results_trace_sweep_highcount.json", "w") as f:
        json.dump({"summary": summary, "n_per_point": N}, f, indent=2)
    print("saved results_trace_sweep_highcount.json")
