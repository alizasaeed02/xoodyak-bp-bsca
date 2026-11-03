"""
Resumable n=100 Keccak sweep for an arbitrary device word size, using the
authors' unmodified bp.py and the same protocol / seed scheme as
run_reproduction_100_random.py (random input) and run_reproduction_100.py
(all-zero input).

Each (sigma, seed) run is appended to a JSONL file as soon as it finishes,
so an interruption loses nothing and re-running skips finished runs.

Outcome per run:
  success -- bp.py ran to completion and errors_in == 0
  failure -- bp.py ran to completion and errors_in != 0
  invalid -- timeout, or bp.py crashed (no LOG line); never counted as failure
"""
import argparse, json, os, re, subprocess, time
from multiprocessing import Pool
from pathlib import Path

import scipy.stats as st

REPO = Path(r"D:\Crypto Reserch\keccaksasca-master\keccak-p")
SHIM = r"D:\Crypto Reserch\bp_crypto\algorithms\keccak_repro\shim"
PYEXE = r"D:\Crypto Reserch\bp_crypto\.venv\Scripts\python.exe"
OUTDIR = Path(__file__).parent / "results_sweep"
TIMEOUT_S = 900
N_PER_SIGMA = 100

LOG_RE = re.compile(r"^LOG ([\d.]+), ([\d.]+), ([0-9A-Fa-f]+), (\w), (\d+), (\d+), (\d+), ([\d.eE+-]+)")

INPUTS = {
    "random": ["--istate", '"RR"*16 + "RR"*(200-16)',
               "--imask", '"00"*16 + "FF"*(200-16)'],
    "zero":   ["--istate", '"RR"*16 + "00"*(200-16)',
               "--imask", '"00"*16 + "FF"*(200-16)'],
}
SIGMAS = {
    "random": [0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0],
    "zero":   [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
}


def seed_for(sigma, k):
    return 8000 + int(sigma * 100) * 10000 + k


def run_one(args):
    scenario, word, sigma, seed = args
    cmd = [PYEXE, "-u", "bp.py", "--name", f"sweep{word}{scenario}",
           "-s", str(sigma), "--seed", f"{seed:08X}",
           "--lanelength", "64", "--rounds", "2",
           *INPUTS[scenario], "--fmask", "0", "-w", str(word)]
    env = dict(os.environ, PYTHONPATH=SHIM)
    rec = {"sigma": sigma, "seed": seed, "word": word, "scenario": scenario}
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           timeout=TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        rec.update(status="invalid", reason=f"timeout after {TIMEOUT_S}s")
        return rec
    for line in p.stdout.splitlines():
        m = LOG_RE.match(line)
        if m:
            errors_in = int(m.group(6))
            rec.update(status="success" if errors_in == 0 else "failure",
                       errors_in=errors_in, errors_out=int(m.group(7)),
                       runtime_s=float(m.group(8)))
            return rec
    tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["no output"]
    rec.update(status="invalid", reason=f"exit {p.returncode}: {tail[0][:300]}")
    return rec


def load_done(path):
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["sigma"], r["seed"])] = r
    return done


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else st.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else st.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def summarise(records, sigmas):
    out = {}
    for s in sigmas:
        rs = [r for r in records if r["sigma"] == s]
        succ = sum(r["status"] == "success" for r in rs)
        fail = sum(r["status"] == "failure" for r in rs)
        inv = sum(r["status"] == "invalid" for r in rs)
        valid = succ + fail
        rate = succ / valid if valid else float("nan")
        lo, hi = clopper_pearson(succ, valid)
        out[str(s)] = {"trials": len(rs), "valid": valid, "invalid": inv,
                       "successes": succ, "failures": fail,
                       "rate": rate, "cp95_lo": lo, "cp95_hi": hi}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["random", "zero"], required=True)
    ap.add_argument("--word", type=int, required=True)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    tag = f"{args.word}bit_{args.scenario}"
    jsonl = OUTDIR / f"runs_{tag}.jsonl"
    sigmas = SIGMAS[args.scenario]
    done = load_done(jsonl)

    jobs = [(args.scenario, args.word, s, seed_for(s, k))
            for s in sigmas for k in range(N_PER_SIGMA)]
    todo = [j for j in jobs if (j[2], j[3]) not in done]
    print(f"[{tag}] {len(jobs)} runs total, {len(done)} already done, "
          f"{len(todo)} to do, workers={args.workers}", flush=True)

    t0 = time.time()
    records = list(done.values())
    reported = set()
    with open(jsonl, "a") as f, Pool(args.workers) as pool:
        for rec in pool.imap_unordered(run_one, todo):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            records.append(rec)
            for s in sigmas:
                if s in reported:
                    continue
                n_done = sum(1 for r in records if r["sigma"] == s)
                if n_done == N_PER_SIGMA:
                    reported.add(s)
                    sm = summarise(records, [s])[str(s)]
                    print(f"[{tag}] sigma={s} done: trials={sm['trials']} "
                          f"valid={sm['valid']} invalid={sm['invalid']} "
                          f"success={sm['successes']} rate={sm['rate']:.3f} "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)

    summary = summarise(records, sigmas)
    (OUTDIR / f"summary_{tag}.json").write_text(json.dumps(
        {"word": args.word, "scenario": args.scenario, "n_per_sigma": N_PER_SIGMA,
         "timeout_s": TIMEOUT_S, "success_criterion": "errors_in == 0",
         "summary": summary}, indent=2))
    print(f"[{tag}] FINISHED in {time.time()-t0:.0f}s -> results_sweep/summary_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
