"""
32-bit random-input sweep with an adaptive stopping rule: for each sigma, run
seeds 1, 2, ... (in order, 2 at a time) until 30 VALID runs are collected,
stopping at 60 seeds per sigma at most. A crash or timeout is invalid and
does not count toward the 30. Uses the authors' unmodified bp.py via
sweep_wordsize.run_one (same command, same success definition as the 8-bit
and 16-bit runs). Results are appended per run, so the script is resumable.
"""
import json, time
from multiprocessing import Pool
from pathlib import Path

from sweep_wordsize import run_one, load_done, summarise

SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
WORD = 32
SCENARIO = "random"
TARGET_VALID = 30
MAX_SEEDS = 60
WORKERS = 2
OUTDIR = Path(__file__).parent / "results_sweep"


def main():
    OUTDIR.mkdir(exist_ok=True)
    jsonl = OUTDIR / f"runs_{WORD}bit_{SCENARIO}.jsonl"
    done = load_done(jsonl)
    t_all = time.time()
    with open(jsonl, "a") as f, Pool(WORKERS) as pool:
        for sigma in SIGMAS:
            t0 = time.time()
            seeds = list(range(1, MAX_SEEDS + 1))
            recs = [r for r in done.values() if r["sigma"] == sigma]
            valid = sum(r["status"] in ("success", "failure") for r in recs)
            next_idx = 0
            while valid < TARGET_VALID and len(recs) < MAX_SEEDS:
                batch = []
                for seed in seeds:
                    if (sigma, seed) in done:
                        continue
                    batch.append((SCENARIO, WORD, sigma, seed))
                    if len(batch) == WORKERS:
                        break
                if not batch:
                    break
                for rec in pool.map(run_one, batch):
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    done[(rec["sigma"], rec["seed"])] = rec
                    recs.append(rec)
                    if rec["status"] in ("success", "failure"):
                        valid += 1
                # stop immediately once target reached (no extra seeds)
                if valid >= TARGET_VALID:
                    break
            sm = summarise(recs, [sigma])[str(sigma)]
            print(f"[32bit_random] sigma={sigma} attempted={len(recs)} valid={sm['valid']} "
                  f"crashed_or_timeout={sm['invalid']} successes={sm['successes']} "
                  f"rate_among_valid={sm['rate']:.3f} "
                  f"cp95=[{sm['cp95_lo']:.3f},{sm['cp95_hi']:.3f}] "
                  f"sigma_elapsed={time.time()-t0:.0f}s total_elapsed={time.time()-t_all:.0f}s",
                  flush=True)
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
