"""Plot: known-nonce D2_BP vs unknown-nonce (order-1), success rate vs sigma,
same style as final_single_trace.png."""
import json
import matplotlib.pyplot as plt

d = json.load(open("unknown_nonce_step3_single_trace_results.json"))
sigmas = sorted(float(s) for s in d["results"])

colors = {"known": "#2a78d6", "unknown": "#e34948"}
labels = {"known": "Known nonce (D2_BP, main method)", "unknown": "Unknown nonce (order-1, 256-bit joint recovery)"}

fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))

rate_k, lo_k, hi_k = [], [], []
rate_u, lo_u, hi_u = [], [], []
for s in sigmas:
    r = d["results"][str(s)]
    nk = r["known_nonce_D2_BP_successes"]; ciK = r["known_nonce_cp95_two_sided"]
    nu = r["unknown_nonce_order1_successes"]; ciU = r["unknown_nonce_cp95_two_sided"]
    rate_k.append(nk / 100); lo_k.append(nk / 100 - ciK[0]); hi_k.append(ciK[1] - nk / 100)
    rate_u.append(nu / 100); lo_u.append(nu / 100 - ciU[0]); hi_u.append(ciU[1] - nu / 100)

ax.errorbar(sigmas, rate_k, yerr=[lo_k, hi_k], fmt="o-", color=colors["known"], lw=2,
            markersize=5, capsize=3, label=labels["known"])
ax.errorbar(sigmas, rate_u, yerr=[lo_u, hi_u], fmt="o-", color=colors["unknown"], lw=2,
            markersize=5, capsize=3, label=labels["unknown"])

ax.set_xlabel(r"$\sigma$")
ax.set_ylabel("Full-key success rate")
ax.set_title("Single-trace: known nonce vs unknown (fully blind) nonce\n"
              "corrected init, n=100/point")
ax.set_ylim(-0.02, 1.05)
ax.grid(alpha=0.25)
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig("unknown_nonce_single_trace.png", dpi=300)
print("saved unknown_nonce_single_trace.png")
