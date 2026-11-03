"""
Extends the 256-bit 8-bit-device random-input sweep with sigma=2.0, and then
sigma=2.25 only if sigma=2.0 has any successes. Same setup, same jsonl file,
same seed scheme as sweep_8bit_random_256.py -- importing its run_one/seed_for
so the two are byte-identical in protocol. Appends only; does not touch the
earlier sigma rows (resumable via load_done, keyed by (sigma, seed), and these
sigmas were never run before).
"""
import json, time
from multiprocessing import Pool
from pathlib import Path

from sweep_wordsize import load_done
from sweep_8bit_random_256 import run_one, seed_for, summarise, N_PER_SIGMA, WORKERS, OUTDIR

JSONL = OUTDIR / "runs_8bit_random_256.jsonl"


def run_sigma(sigma, done, f, pool):
    jobs = [(sigma, seed_for(sigma, k)) for k in range(N_PER_SIGMA)]
    todo = [j for j in jobs if j not in done]
    print(f"[8bit_random_256_extend] sigma={sigma}: {len(jobs)} runs total, "
          f"{len(jobs)-len(todo)} already done, {len(todo)} to do", flush=True)
    t0 = time.time()
    records = [r for r in done.values() if r["sigma"] == sigma]
    for rec in pool.imap_unordered(run_one, todo):
        f.write(json.dumps(rec) + "\n")
        f.flush()
        done[(rec["sigma"], rec["seed"])] = rec
        records.append(rec)
    sm = summarise(records, [sigma])[str(sigma)]
    print(f"[8bit_random_256_extend] sigma={sigma} done: trials={sm['trials']} "
          f"valid={sm['valid']} invalid={sm['invalid']} success={sm['successes']} "
          f"rate={sm['rate']:.3f} elapsed={time.time()-t0:.0f}s", flush=True)
    return sm


if __name__ == "__main__":
    done = load_done(JSONL)
    with open(JSONL, "a") as f, Pool(WORKERS) as pool:
        sm20 = run_sigma(2.0, done, f, pool)
        if sm20["successes"] > 0:
            print("[8bit_random_256_extend] sigma=2.0 had successes -> running sigma=2.25", flush=True)
            run_sigma(2.25, done, f, pool)
        else:
            print("[8bit_random_256_extend] sigma=2.0 had 0 successes -> skipping sigma=2.25 per instructions", flush=True)
    print("[8bit_random_256_extend] FINISHED", flush=True)
