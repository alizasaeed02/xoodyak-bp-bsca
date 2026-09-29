"""
Regenerates xoodyak_multitrace.png from results.json's actual data (both
base_grid, 1-50 traces, and extended_high_trace_count_check, 55-200 traces),
covering the full tested trace range and labelling each sigma with the exact
value from results.json's smallest_confirmed_100pct_trace_count field.
"""
import json
import matplotlib.pyplot as plt

d = json.load(open("results.json"))["multi_trace"]
base = d["base_grid"]
high = d["extended_high_trace_count_check"]
crossover = d["smallest_confirmed_100pct_trace_count"]

sigmas = [0.5, 1.0, 1.5, 2.0, 2.25]
colors = {0.5: "#2a78d6", 1.0: "#eda100", 1.5: "#e34948", 2.0: "#7a4fd6", 2.25: "#1baf7a"}

plt.figure(figsize=(8, 6))
for sigma in sigmas:
    points = {}
    for k, v in base.items():
        s, t = k.rsplit("_", 1)
        if float(s) == sigma:
            points[int(t)] = v["rate"]
    for k, v in high.items():
        s, t = k.rsplit("_", 1)
        if float(s) == sigma:
            points[int(t)] = v["rate"]
    traces = sorted(points)
    rates = [points[t] for t in traces]
    plt.plot(traces, rates, "o-", color=colors[sigma], lw=2, markersize=5,
              label=fr"$\sigma$={sigma}  (100% at {crossover[str(sigma)]} traces)")

plt.xlabel("number of traces actually tested (repeated measurements, same key)")
plt.ylabel("full 128-bit key recovery rate")
plt.title("Xoodyak: full key recovery vs. trace count\n(all points from real executed runs, up to 200 traces)")
plt.ylim(-0.02, 1.05)
plt.xscale("log")
plt.xticks([1, 5, 10, 20, 30, 50, 75, 100, 150, 200], [1, 5, 10, 20, 30, 50, 75, 100, 150, 200])
plt.grid(alpha=0.3, which="both")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("xoodyak_multitrace.png", dpi=200)

print("x-axis range: 1 to", max(int(k.rsplit('_',1)[1]) for k in {**base, **high}))
print("crossover values plotted:", crossover)
print("saved xoodyak_multitrace.png")
