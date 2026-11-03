"""
8-bit device, random-input, 256-bit secret sweep. Authors' unmodified bp.py,
inst_generic_256_rand.sh settings (istate R, imask '"00"*32 + "FF"*(200-32)'),
same success definition / BP settings as the 128-bit sweeps (sweep_wordsize.py):
errors_in == 0, defaults for iterations/damping/clustersize.

sigma = 0.5, 0.75, 1.0, 1.25, 1.5, 1.75; n=100/sigma; 2 workers; timeout 900s.
Crash or timeout = invalid, excluded from the rate, not a failure.
Resumable: appends to results_sweep/runs_8bit_random_256.jsonl, skips done runs.
"""
import json, os, re, subprocess, time
from multiprocessing import Pool
from pathlib import Path

from sweep_wordsize import clopper_pearson, load_done

REPO = Path(r"D:\Crypto Reserch\keccaksasca-master\keccak-p")
SHIM = r"D:\Crypto Reserch\bp_crypto\algorithms\keccak_repro\shim"
PYEXE = r"D:\Crypto Reserch\bp_crypto\.venv\Scripts\python.exe"
OUTDIR = Path(__file__).parent / "results_sweep"
TIMEOUT_S = 900
N_PER_SIGMA = 100
WORKERS = 2
SIGMAS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

LOG_RE = re.compile(r"^LOG ([\d.]+), ([\d.]+), ([0-9A-Fa-f]+), (\w), (\d+), (\d+), (\d+), ([\d.eE+-]+)")

ISTATE = "R"
IMASK = '"00"*32 + "FF"*(200-32)'


def seed_for(sigma, k):
    return 8000 + int(sigma * 100) * 10000 + k


def run_one(args):
    sigma, seed = args
    cmd = [PYEXE, "-u", "bp.py", "--name", "sweep8rand256", "-s", str(sigma),
           "--seed", f"{seed:08X}", "--lanelength", "64", "--rounds", "2",
           "--istate", ISTATE, "--imask", IMASK, "--fmask", "0", "-w", "8"]
    env = dict(os.environ, PYTHONPATH=SHIM)
    rec = {"sigma": sigma, "seed": seed, "word": 8, "scenario": "random_256"}
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
    OUTDIR.mkdir(exist_ok=True)
    jsonl = OUTDIR / "runs_8bit_random_256.jsonl"
    done = load_done(jsonl)

    jobs = [(s, seed_for(s, k)) for s in SIGMAS for k in range(N_PER_SIGMA)]
    todo = [j for j in jobs if j not in done]
    print(f"[8bit_random_256] {len(jobs)} runs total, {len(done)} already done, "
          f"{len(todo)} to do, workers={WORKERS}", flush=True)

    t0 = time.time()
    records = list(done.values())
    reported = set()
    with open(jsonl, "a") as f, Pool(WORKERS) as pool:
        for rec in pool.imap_unordered(run_one, todo):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            records.append(rec)
            for s in SIGMAS:
                if s in reported:
                    continue
                n_done = sum(1 for r in records if r["sigma"] == s)
                if n_done == N_PER_SIGMA:
                    reported.add(s)
                    sm = summarise(records, [s])[str(s)]
                    print(f"[8bit_random_256] sigma={s} done: trials={sm['trials']} "
                          f"valid={sm['valid']} invalid={sm['invalid']} "
                          f"success={sm['successes']} rate={sm['rate']:.3f} "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)

    summary = summarise(records, SIGMAS)
    (OUTDIR / "summary_8bit_random_256.json").write_text(json.dumps(
        {"word": 8, "scenario": "random_256", "secret_bits": 256,
         "n_per_sigma": N_PER_SIGMA, "timeout_s": TIMEOUT_S,
         "success_criterion": "errors_in == 0", "summary": summary}, indent=2))
    print(f"[8bit_random_256] FINISHED in {time.time()-t0:.0f}s -> "
          f"results_sweep/summary_8bit_random_256.json", flush=True)


if __name__ == "__main__":
    main()
