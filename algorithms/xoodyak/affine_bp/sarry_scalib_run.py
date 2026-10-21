"""
Runner for the SCALib Sarry-style byte-rank experiment (sarry_scalib_core.py).

  python sarry_scalib_run.py pilot          noise-free check + 50-key pilot,
                                            iteration counts 30 and 100 (plus a
                                            diagnostic scan), versions A and B
  python sarry_scalib_run.py grid <iters>   1000 keys x 7 sigma x versions A/B,
                                            resumable (one JSON line per trial)

Seeds: seed = 3000 + round(sigma*100)*1000 + k  (k = 0..n-1); key from
default_rng(seed), nonce from default_rng(seed+500_000_000), leakage noise
from default_rng(seed+1_000_000_000). The pilot keys are the first 50 keys of
the grid. Rank is reported 1-indexed (1 = best); ties get the average rank,
so uniform marginals give 128.5. top-k counts ranks <= k.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sarry_scalib_core import run_attack

HERE = Path(__file__).parent
GRID_SIGMAS = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00]
NOISE_FREE_SIGMA = 1e-3


def seed_for(sigma, k):
    return 3000 + int(round(sigma * 100)) * 1000 + k


def job(args):
    sigma, version, n_iters, k = args
    seed = seed_for(sigma, k)
    t0 = time.perf_counter()
    ranks, nan_flag = run_attack(seed, sigma, version, n_iters)
    return {"sigma": sigma, "version": version, "n_iters": n_iters, "k": k, "seed": seed,
            "ranks0": ranks, "nan": nan_flag, "runtime_s": time.perf_counter() - t0}


def summarize(rows):
    valid = [r for r in rows if not r["nan"]]
    ranks = np.array([x for r in valid for x in r["ranks0"]], dtype=float) + 1.0
    out = {"trials": len(rows), "nan_trials": len(rows) - len(valid), "valid_trials": len(valid),
           "mean_runtime_s_per_key": float(np.mean([r["runtime_s"] for r in rows]))}
    if len(ranks):
        out.update(mean_rank=float(ranks.mean()), std_rank=float(ranks.std()),
                   top1_pct=float(100 * np.mean(ranks <= 1)), top5_pct=float(100 * np.mean(ranks <= 5)),
                   top10_pct=float(100 * np.mean(ranks <= 10)),
                   full_key=int(sum(all(x == 0 for x in r["ranks0"]) for r in valid)))
    return out


def fmt(s):
    if "mean_rank" not in s:
        return f"ALL NaN ({s['nan_trials']}/{s['trials']})"
    return (f"mean={s['mean_rank']:7.2f} std={s['std_rank']:6.2f} top1={s['top1_pct']:5.2f}% "
            f"top5={s['top5_pct']:5.2f}% top10={s['top10_pct']:5.2f}% full_key={s['full_key']}/{s['valid_trials']} "
            f"NaN={s['nan_trials']}/{s['trials']} {s['mean_runtime_s_per_key']:.2f}s/key")


def pilot():
    out = {"noise_free": {}, "pilot": {}, "iteration_scan": {}}
    t_all = time.time()
    with Pool(4) as pool:
        for version in ("A", "B"):
            for n_iters in (30, 100):
                rows = pool.map(job, [(NOISE_FREE_SIGMA, version, n_iters, k) for k in range(50)])
                s = summarize(rows)
                out["noise_free"][f"{version}_{n_iters}"] = {"summary": s, "trials": rows}
                print(f"noise-free sigma={NOISE_FREE_SIGMA} ver={version} iters={n_iters}: {fmt(s)}", flush=True)
        for version in ("A", "B"):
            for sigma in (0.10, 0.50):
                for n_iters in (5, 10, 20, 30, 50, 100):
                    rows = pool.map(job, [(sigma, version, n_iters, k) for k in range(50)])
                    s = summarize(rows)
                    dest = "pilot" if n_iters in (30, 100) else "iteration_scan"
                    out[dest][f"{version}_{sigma}_{n_iters}"] = {"summary": s, "trials": rows}
                    print(f"{dest:14s} ver={version} sigma={sigma} iters={n_iters:3d}: {fmt(s)}", flush=True)
    out["elapsed_s"] = time.time() - t_all
    (HERE / "sarry_scalib_pilot_results.json").write_text(json.dumps(out, indent=1))
    print(f"pilot elapsed {out['elapsed_s']:.0f}s -> sarry_scalib_pilot_results.json", flush=True)


def grid(n_iters, n_keys=1000):
    path = HERE / f"sarry_scalib_grid_iters{n_iters}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["sigma"], r["version"], r["k"]))
    t_all = time.time()
    with open(path, "a") as f, Pool(4) as pool:
        for version in ("A", "B"):
            for sigma in GRID_SIGMAS:
                todo = [(sigma, version, n_iters, k) for k in range(n_keys) if (sigma, version, k) not in done]
                t0 = time.time()
                for r in pool.imap_unordered(job, todo, chunksize=4):
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                print(f"GRID ver={version} sigma={sigma}: {len(todo)} new trials in {time.time()-t0:.0f}s "
                      f"(total {time.time()-t_all:.0f}s)", flush=True)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    summary = {}
    for version in ("A", "B"):
        for sigma in GRID_SIGMAS:
            sel = [r for r in rows if r["version"] == version and r["sigma"] == sigma]
            summary[f"{version}_{sigma}"] = summarize(sel)
            print(f"SUMMARY ver={version} sigma={sigma}: {fmt(summary[f'{version}_{sigma}'])}", flush=True)
    (HERE / f"sarry_scalib_grid_iters{n_iters}_summary.json").write_text(json.dumps(
        {"n_iters": n_iters, "n_keys": n_keys, "sigmas": GRID_SIGMAS, "elapsed_this_run_s": time.time() - t_all,
         "random_guess_reference": {"mean_rank": 128.5, "top1_pct": 0.39, "top5_pct": 1.95, "top10_pct": 3.91},
         "summary": summary}, indent=1))
    print(f"GRID FINISHED in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "pilot":
        pilot()
    else:
        grid(int(sys.argv[2]))
