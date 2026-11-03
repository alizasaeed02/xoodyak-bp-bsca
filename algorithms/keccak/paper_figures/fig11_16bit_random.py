"""
fig11_16bit_random.py
=====================================================================
16-bit-device panel, random public input, matching the exact style of
keccak_reproduction.png (color, marker, fonts, panel size, title
format, suptitle wording) used in the paper for the 8-bit figure, with
95% Clopper-Pearson error bars added (the 8-bit figure has none; this
is the only intentional difference).

Data source: results_sweep/summary_16bit_random.json, produced by
sweep_wordsize.py from the authors' unmodified bp.py, n=100 trials/sigma.
=====================================================================
"""
import json
import matplotlib.pyplot as plt

SRC = r"D:\Crypto Reserch\bp_crypto\algorithms\keccak_repro\results_sweep\summary_16bit_random.json"

d = json.load(open(SRC))
summary = d["summary"]
sigmas = sorted(float(s) for s in summary)


def get(s):
    for k, v in summary.items():
        if abs(float(k) - s) < 1e-9:
            return v
    raise KeyError(s)


rate = [get(s)["rate"] for s in sigmas]
lo = [get(s)["cp95_lo"] for s in sigmas]
hi = [get(s)["cp95_hi"] for s in sigmas]
yerr_lo = [r - l for r, l in zip(rate, lo)]
yerr_hi = [h - r for h, r in zip(hi, rate)]

fig, ax = plt.subplots(1, 1, figsize=(6, 4.6))
ax.errorbar(sigmas, rate, yerr=[yerr_lo, yerr_hi], fmt="s-", color="#2a78d6",
            lw=2, markersize=6, capsize=3, label="16-bit device, random input (our reproduction)")
ax.set_xlabel(r"$\sigma$")
ax.set_ylabel("Success Rate")
ax.set_title("16-bit device, random public input")
ax.set_xlim(0, 3.1)
ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()
fig.savefig("fig11_16bit_random.png", dpi=300)
print("saved fig11_16bit_random.png")
for s in sigmas:
    g = get(s)
    print(f"sigma={s}: valid={g['valid']} invalid={g['invalid']} "
          f"successes={g['successes']} rate={g['rate']:.4f} "
          f"cp95=[{g['cp95_lo']:.4f},{g['cp95_hi']:.4f}]")
