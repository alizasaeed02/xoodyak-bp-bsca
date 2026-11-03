"""
8-bit device, random public input: 128-bit secret vs 256-bit secret, same
style as the existing 8-bit panel (color/marker/fonts/figsize/grid), with a
second curve and colour for the 256-bit secret.

Sources:
  128-bit: results/fig4a_repro_100.json (n=100/sigma, sigma 0.5-3.0)
  256-bit: results_sweep/summary_8bit_random_256.json (n=100/sigma, sigma 0.5-1.75)
"""
import sys
import json
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\Crypto Reserch\bp_crypto\algorithms\keccak_repro")
from sweep_wordsize import clopper_pearson

d128 = json.load(open(r"D:\Crypto Reserch\bp_crypto\algorithms\keccak_repro\results\fig4a_repro_100.json"))["summary"]
d256 = json.load(open(r"D:\Crypto Reserch\bp_crypto\algorithms\keccak_repro\results_sweep\summary_8bit_random_256.json"))["summary"]

sig128 = sorted(float(s) for s in d128)
rate128, lo128, hi128 = [], [], []
for s in sig128:
    v = d128[str(s) if str(s) in d128 else s]
    valid = v["n_valid"]
    succ = round(v["rate"] * valid)
    lo, hi = clopper_pearson(succ, valid)
    rate128.append(v["rate"]); lo128.append(lo); hi128.append(hi)

sig256 = sorted(float(s) for s in d256)
rate256 = [d256[str(s)]["rate"] for s in sig256]
lo256 = [d256[str(s)]["cp95_lo"] for s in sig256]
hi256 = [d256[str(s)]["cp95_hi"] for s in sig256]

yerr128 = [[r - l for r, l in zip(rate128, lo128)], [h - r for h, r in zip(hi128, rate128)]]
yerr256 = [[r - l for r, l in zip(rate256, lo256)], [h - r for h, r in zip(hi256, rate256)]]

fig, ax = plt.subplots(1, 1, figsize=(6, 4.6))
ax.errorbar(sig128, rate128, yerr=yerr128, fmt="s-", color="#2a78d6", lw=2,
            markersize=6, capsize=3, label="128-bit secret")
ax.errorbar(sig256, rate256, yerr=yerr256, fmt="s-", color="#e34948", lw=2,
            markersize=6, capsize=3, label="256-bit secret")
ax.set_xlabel(r"$\sigma$")
ax.set_ylabel("Success Rate")
ax.set_title("8-bit device, random public input")
ax.set_xlim(0, 3.1)
ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()
fig.savefig("fig_128_vs_256_8bit_random.png", dpi=300)
print("saved fig_128_vs_256_8bit_random.png")
print("128-bit:", dict(zip(sig128, rate128)))
print("256-bit:", dict(zip(sig256, rate256)))
