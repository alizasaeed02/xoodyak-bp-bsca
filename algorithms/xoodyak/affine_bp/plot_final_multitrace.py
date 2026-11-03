"""Plot: D0 (BP, fixed rows) vs D2_BP (main) vs D2_exact (comparison),
one panel per sigma, success rate vs number of traces, from
final_multitrace_grid_results.json."""
import json
import matplotlib.pyplot as plt

d = json.load(open("final_multitrace_grid_results.json"))
sigmas = d["sigmas"]

fig, axes = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 4.4), sharey=True)
for ax, sigma in zip(axes, sigmas):
    res = d["results"][str(sigma)]
    points = res["points"]
    ns = sorted(int(n) for n in points)
    d0 = [points[str(n)]["D0_successes"] / points[str(n)]["n"] for n in ns]
    d2bp = [points[str(n)]["D2_BP_successes"] / points[str(n)]["n"] for n in ns]
    d2e = [points[str(n)]["D2_exact_successes"] / points[str(n)]["n"] for n in ns]
    ax.plot(ns, d0, "o-", color="#888888", lw=2, markersize=4, label="D0 (BP, fixed rows)")
    ax.plot(ns, d2bp, "o-", color="#e34948", lw=2, markersize=4, label="D2_BP (main)")
    ax.plot(ns, d2e, "o-", color="#2a78d6", lw=2, markersize=4, label="D2_exact")
    title = fr"$\sigma$={sigma}"
    if res["first_N_100pct_D2_BP"]:
        title += f"\nD2_BP 100% @ N={res['first_N_100pct_D2_BP']}"
    if res["first_N_100pct_D0"]:
        title += f", D0 100% @ N={res['first_N_100pct_D0']}"
    else:
        title += ", D0 never reached 100%"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("number of traces (N)")
    ax.set_xscale("log")
    ax.grid(alpha=0.25)

axes[0].set_ylabel("Full-key success rate")
axes[0].set_ylim(-0.02, 1.05)
axes[-1].legend(fontsize=7, loc="lower right")
fig.suptitle("Multi-trace: " + d["observation_model"][:90] + "...", fontsize=8)
fig.tight_layout()
fig.savefig("final_multitrace.png", dpi=300)
print("saved final_multitrace.png")
