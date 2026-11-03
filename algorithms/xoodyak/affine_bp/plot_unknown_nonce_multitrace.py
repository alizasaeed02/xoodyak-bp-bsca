"""Plot: unknown-nonce multi-trace, one panel per sigma, success rate vs N,
same style as final_multitrace.png, with the known-nonce D2_BP
two-consecutive threshold marked as a vertical reference line."""
import json
import matplotlib.pyplot as plt

d = json.load(open("unknown_nonce_step4_multitrace_results.json"))
sigmas = d["sigmas"]
known_thresholds = d["known_nonce_D2_BP_first_two_consecutive_for_comparison"]

fig, axes = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 4.4), sharey=True)
for ax, sigma in zip(axes, sigmas):
    res = d["results"][str(sigma)]
    points = res["points"]
    ns = sorted(int(n) for n in points)
    rates = [points[str(n)]["successes"] / points[str(n)]["n"] for n in ns]
    ax.plot(ns, rates, "o-", color="#e34948", lw=2, markersize=4, label="Unknown nonce (order-1)")
    kt = known_thresholds.get(str(sigma))
    if kt is not None:
        ax.axvline(kt, color="#2a78d6", lw=1.5, linestyle="--", label=f"Known-nonce D2_BP threshold (N={kt})")
    title = fr"$\sigma$={sigma}"
    if res["first_two_consecutive_100pct"]:
        title += f"\nunknown-nonce 100% @ N={res['first_two_consecutive_100pct']}"
    else:
        title += "\nunknown-nonce never reached 100% (in tested range)"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("number of traces (N)")
    ax.set_xscale("log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")

axes[0].set_ylabel("Full-key success rate")
axes[0].set_ylim(-0.02, 1.05)
fig.suptitle("Unknown-nonce multi-trace (key AND nonce fixed across N traces, attacker never uses true nonce)")
fig.tight_layout()
fig.savefig("unknown_nonce_multitrace.png", dpi=300)
print("saved unknown_nonce_multitrace.png")
