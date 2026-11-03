"""Plot: D0 (BP, fixed rows) vs D2_BP (main method) vs D2_exact (comparison),
success rate vs sigma, from final_single_trace_grid_results.json."""
import json
import matplotlib.pyplot as plt

d = json.load(open("final_single_trace_grid_results.json"))
sigmas = sorted(float(s) for s in d["results"])

colors = {"D0": "#888888", "D2_BP": "#e34948", "D2_exact": "#2a78d6"}
labels = {"D0": "D0 (BP, existing fixed-row decoder -- published pipeline)",
          "D2_BP": "D2_BP (BP + reliability decoder -- main method)",
          "D2_exact": "D2_exact (exact inference + reliability decoder)"}

fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
for method in ["D0", "D2_BP", "D2_exact"]:
    rates = [d["results"][str(s)]["summary"][method]["rate"] for s in sigmas]
    los, his = [], []
    for s in sigmas:
        lo, hi = d["results"][str(s)]["summary"][method]["two_sided_95ci"]
        r = d["results"][str(s)]["summary"][method]["rate"]
        los.append(r - lo); his.append(hi - r)
    ax.errorbar(sigmas, rates, yerr=[los, his], fmt="o-", color=colors[method],
                lw=2, markersize=5, capsize=3, label=labels[method])

ax.set_xlabel(r"$\sigma$")
ax.set_ylabel("Full-key success rate")
ax.set_title("Single-trace: published pipeline (D0) vs reliability decoder (D2)\n"
              "corrected init, random public nonce per trial, n=100/point")
ax.set_ylim(-0.02, 1.05)
ax.grid(alpha=0.25)
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig("final_single_trace.png", dpi=300)
print("saved final_single_trace.png")
