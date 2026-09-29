# Xoodyak full key recovery — corrected chi-only + affine-decode design

Implements belief propagation restricted to Xoodyak's nonlinear $\chi$ step
on the **real** 4×3×32 Xoodoo state (not the byte-level ring approximation
in `../scalib/`), with $\theta \circ \rho_{west} \circ \iota$ solved as an
**exact GF(2) affine map** instead of being folded into the BP graph. See
`linear_theta.py` for why: because Xoodyak's key-load pads the non-key
planes with zero, that composition is a fixed public linear map from key
to chi-input, invertible by direct linear algebra — putting it inside BP
only adds unhelpful cycles.

Both the permutation logic and the round-constant indexing are
cross-validated bit-for-bit against an independent C++ reference
implementation, [itzmeanjan/xoodyak](https://github.com/itzmeanjan/xoodyak)
(`include/xoodoo.hpp`) — an implementation written independently of this
project, not by us — across the full 12-round permutation, on the
all-zero state and 30 random states. This is `validate_against_cpp_reference()`
at the bottom of `linear_theta.py`; it prints PASS/FAIL per check, not a
silent assert, and running `python linear_theta.py` reproduces:

```
self_test OK: affine map + GF(2) decode exact on 100 random keys

[all-zero state]      match: PASS
[30 random states]  match: 30/30  PASS

OVERALL: PASS -- ref.xoodoo() matches the independent C++ reference (xoodyak-master/include/xoodoo.hpp) on the full 12-round permutation.
```

## Config

- `N_ITERS = 30` loopy BP iterations (`chi_bit_graph.py`)
- Leakage: per-bit noisy Hamming weight, `L = bit + N(0, sigma)`
- Success: all 128 key bits recovered exactly

## Results, and an honest correction

- `singletrace_results.json` — full 128-bit key recovery, single trace,
  sigma in {0.05, ..., 0.45, 0.5, 0.6, 0.7, 0.8, 1.0}, **n=100/point**.
- `multitrace_results.json` — recovery rate vs. trace count, 1-50 traces,
  sigma in {0.5, 0.75, 1.0, 1.5, 2.0}, **n=100/point**.
- `results_trace_sweep_highcount.json` (in the source repo this was copied
  from) / merged into `results.json` — extended check at 55-200 traces,
  sigma in {1.0, 1.5, 2.0, 2.25}, **n=50/point**, run specifically because
  an earlier ad-hoc n=50 spot-check gave an over-optimistic first estimate
  of the sigma=2.0/2.25 crossover (75 and 100 traces) that this larger,
  properly saved run does not confirm.

**Smallest trace count with a confirmed clean 100% recovery rate:**

| sigma | traces | basis |
|---|---|---|
| 0.5  | 7   | n=100 |
| 1.0  | 25–30 | 100% first seen at 25 traces (n=50); also confirmed at 30 traces (n=100, the more statistically robust point — 20 traces only reaches 97%) |
| 1.5  | 60  | n=50, consistent with 99% already at 50 traces (n=100) |
| 2.0  | 150 | n=50; 75 traces only reaches 94%, 100 traces only reaches 98% |
| 2.25 | 150 | n=50; 100 traces only reaches 96% |

The sigma=2.0/2.25 numbers are the correction: if you've seen "75 traces"
or "100 traces" quoted anywhere else for these two noise levels, that was
from the earlier, smaller, less reliable check — treat `results.json` in
this folder as the authoritative numbers.

## Files

- `linear_theta.py` — the GF(2) affine map for theta/rho_west/iota, plus
  self-test and the independent C++ cross-validation
- `chi_bit_graph.py` — bit-level chi-only BP factor graph, plus self-test
- `run_full_key_sweep.py` — single-trace sweep driver
- `run_trace_sweep.py` — multi-trace sweep driver (1-50 traces)
- `run_trace_sweep_highcount.py` — extended multi-trace driver (55-200 traces)
- `results.json` — merged, final results (single-trace + both multi-trace runs)
- `xoodyak_singletrace.png`, `xoodyak_multitrace.png` — plots
