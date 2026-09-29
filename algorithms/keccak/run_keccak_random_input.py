"""
run_reproduction_100_random.py
=====================================================================
Fig. 4a reproduction: 8-bit device, RANDOM public input, 128-bit
secret. Same protocol as run_reproduction_100.py (their own code,
n=100 trials/sigma), but the non-secret 256 bits of the state are
random-but-known (istate = all 'R', imask marks the first 128 bits
unknown / rest known) instead of fixed to zero.
=====================================================================
"""
import os, re, subprocess, time, json
from pathlib import Path
from multiprocessing import Pool

REPO = Path(r"D:\Crypto Reserch\keccaksasca-master\keccak-p")
SHIM = r"D:\Crypto Reserch\bp_crypto\algorithms\keccak_repro\shim"
PYEXE = r"D:\Crypto Reserch\bp_crypto\.venv\Scripts\python.exe"

LOG_RE = re.compile(r"^LOG ([\d.]+), ([\d.]+), ([0-9A-Fa-f]+), (\w), (\d+), (\d+), (\d+), ([\d.eE+-]+)")

COMMON_ARGS = [
    "--lanelength", "64",
    "--rounds", "2",
    "--istate", '"RR"*16 + "RR"*(200-16)',   # random public input (vs "00"*... for all-zero)
    "--imask", '"00"*16 + "FF"*(200-16)',
    "--fmask", "0",
    "-w", "8",
]

TIMEOUT_S = 900


def run_one(args):
    sigma, seed = args
    cmd = [PYEXE, "-u", "bp.py", "--name", "repro100rand", "-s", str(sigma), "--seed", f"{seed:08X}"] + COMMON_ARGS
    env = dict(os.environ)
    env["PYTHONPATH"] = SHIM
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return (sigma, seed, None)
    for line in p.stdout.splitlines():
        m = LOG_RE.match(line)
        if m:
            errors_in = int(m.group(6))
            return (sigma, seed, errors_in == 0)
    return (sigma, seed, None)


if __name__ == "__main__":
    SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0]
    N_PER_SIGMA = 100
    jobs = [(s, 8000 + int(s * 100) * 10000 + k) for s in SIGMAS for k in range(N_PER_SIGMA)]

    print(f"{len(jobs)} total runs, sigma in {SIGMAS}, {N_PER_SIGMA} seeds each, timeout={TIMEOUT_S}s, 4 workers")
    t0 = time.time()
    results = {s: [] for s in SIGMAS}
    done = 0
    with Pool(4) as pool:
        for sigma, seed, ok in pool.imap_unordered(run_one, jobs):
            done += 1
            results[sigma].append(ok)
            if done % 25 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                print(f"[{done}/{len(jobs)}] {elapsed:.0f}s elapsed", flush=True)

    print("\nsigma | success_rate | n_valid | n_timeout")
    summary = {}
    for s in SIGMAS:
        vals = results[s]
        valid = [v for v in vals if v is not None]
        timeout_n = len(vals) - len(valid)
        rate = sum(valid) / len(valid) if valid else float("nan")
        summary[s] = {"rate": rate, "n_valid": len(valid), "n_timeout": timeout_n}
        print(f"{s:>5} | {rate:>12.3f} | {len(valid):>7} | {timeout_n:>9}")

    with open(Path(__file__).parent / "results" / "fig4a_repro_100.json", "w") as f:
        json.dump({"summary": summary, "sigmas": SIGMAS, "n_per_sigma": N_PER_SIGMA, "timeout": TIMEOUT_S}, f, indent=2)
    print("saved results/fig4a_repro_100.json")
