"""
fig11_32bit_random.py
=====================================================================
32-bit-device panel, random public input, same style/legend/axis
conventions as fig11_16bit_random.py (which matches the 8-bit
keccak_reproduction.png panel): color #2a78d6, "s-" markers, lw=2,
markersize=6, default fonts, figsize=(6,4.6), grid(alpha=0.25), no
top suptitle, x up to 3.1 so the last point/error bar isn't clipped.

Data source: results_sweep/runs_32bit_random.jsonl (adaptive sweep:
seeds run until 30 valid or 60 seeds per sigma; a crash/timeout is
invalid, excluded from the rate, not counted as a failure).
=====================================================================
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Crypto Reserch\bp_crypto\algorithms\keccak_repro")
from sweep_wordsize import summarise

import matplotlib.pyplot as plt

SRC = Path(r"D:\Crypto Reserch\bp_crypto\algorithms\keccak_repro\results_sweep\runs_32bit_random.jsonl")

recs = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
sigmas = sorted(set(r["sigma"] for r in recs))
summary = summarise(recs, sigmas)

rate = [summary[str(s)]["rate"] for s in sigmas]
lo = [summary[str(s)]["cp95_lo"] for s in sigmas]
hi = [summary[str(s)]["cp95_hi"] for s in sigmas]
yerr_lo = [r - l for r, l in zip(rate, lo)]
yerr_hi = [h - r for h, r in zip(hi, rate)]

fig, ax = plt.subplots(1, 1, figsize=(6, 4.6))
ax.errorbar(sigmas, rate, yerr=[yerr_lo, yerr_hi], fmt="s-", color="#2a78d6",
            lw=2, markersize=6, capsize=3, label="32-bit device, random input (our reproduction)")
ax.set_xlabel(r"$\sigma$")
ax.set_ylabel("Success Rate")
ax.set_title("32-bit device, random public input")
ax.set_xlim(0, 3.1)
ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.25)
ax.legend(loc="lower left")

fig.tight_layout()
fig.savefig("fig11_32bit_random.png", dpi=300)
print("saved fig11_32bit_random.png")
for s in sigmas:
    g = summary[str(s)]
    print(f"sigma={s}: attempted={g['trials']} valid={g['valid']} invalid={g['invalid']} "
          f"successes={g['successes']} rate={g['rate']:.4f} "
          f"cp95=[{g['cp95_lo']:.4f},{g['cp95_hi']:.4f}]")
