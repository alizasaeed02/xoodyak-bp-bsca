# Blind Side-Channel Attack on Xoodyak Using Belief Propagation

Code release accompanying the paper *"Blind Side-Channel Attack on Xoodyak Using
Belief Propagation"*, including the Elephant and Sparkle reproductions used for
cross-cipher comparison.

## Setup

```
pip install -r requirements.txt
```

Each script is run from its own directory:

```
cd algorithms/xoodyak/scalib && python xoodyak_scalib_attack.py
cd algorithms/elephant       && python run_experiment.py
cd algorithms/sparkel        && python run_experiment.py
```

---

## Xoodyak — Table 4

**Script:** `algorithms/xoodyak/scalib/xoodyak_scalib_attack.py`
(self-contained; depends only on `numpy` and `scalib`).

Attacks the χ (chi) step of Xoodyak's Xoodoo permutation at byte granularity
(256-value candidates), using SCALib's `FactorGraph`/`BPState` belief-propagation
engine. Config (top of file): `N_BYTES=20, EXPERIMENTS=1000, BP_ITER=20`, run
across 11 noise levels σ ∈ {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 1.0}.

Re-running overwrites `scalib_ranks.npy`, `scalib_success.npy`, `scalib_timing.json`
in place with fresh random keys — exact numbers will vary trial-to-trial, but
aggregate statistics match the included results closely at 1000 trials.

**Included results** (already at the paper's 1000-trial count):
- `scalib_ranks.npy` — dict keyed by σ, each value a length-20000 (1000 trials × 20 bytes) int array of per-(key byte, trial) ranks (0 = correct byte ranked first out of 256).
- `scalib_success.npy` — same shape, 0/1 full-key-recovery flags per trial.
- `scalib_timing.json` — real per-σ wall-clock seconds for the included run (`time.perf_counter()`), e.g. σ=0.1 → 25.86 s / 1000 trials.

Top-1/5/10 success rate, computed from `scalib_ranks.npy` (`top-k = mean(rank < k)`):

| σ | top-1 % | top-5 % | top-10 % | mean rank |
|---|---|---|---|---|
| 0.10 | 25.52 | 44.16 | 56.02 | 13.31 |
| 0.15 | 19.82 | 39.07 | 53.62 | 14.29 |
| 0.20 | 19.24 | 38.70 | 52.25 | 14.78 |
| 0.30 | 17.07 | 35.14 | 48.68 | 18.43 |
| 0.50 | 12.45 | 26.38 | 37.81 | 29.49 |
| 0.70 | 11.13 | 22.96 | 32.34 | 39.15 |
| 1.00 | 7.67 | 16.78 | 24.86 | 52.64 |

---

## Elephant — Table 2

**Script:** `algorithms/elephant/run_experiment.py`
**Actual import chain (verified):** `run_experiment.py` → `elephant_attack.py` →
`dumbo_lfsr.py`, `byte_bp_engine.py`. All `numpy`-only.

`byte_bp_engine.py` is a **byte-level** BP engine (256-value variables, not bit-
level): the LFSR's 3-tap feedback (`x0`, `x3`, `x13`) is decomposed into two
chained 2-input factors (`make_intermediate_factor`, then `make_final_factor`),
with no carry chains, per the module's own docstring. This is the engine that
produced every number in the paper's Table 2 and the structural comparison in
the Comparative Analysis section — an earlier, separate exploratory file,
`elephant_bit_bp.py`, was considered during drafting but is **not** part of this
pipeline and is intentionally excluded from this release.

Config: σ ∈ {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 1.0},
`N_EXTRA_BYTES=20`, `N_ITERS=15` (fixed, no early stopping).

**⚠️ Trial-count note:** `N_TRIALS` in this copy is set to **1000** (for
cross-cipher consistency with Xoodyak and Sparkle). The included `results.json`
was **not regenerated** — it still holds the original **100-trial** run
(`results.json`'s own `config.n_trials` field correctly reports `100`).
Re-running fresh at 1000 trials will produce tighter tail statistics; the
mean-rank trend across σ should match closely.

---

## Sparkle — Table 3

**Script:** `algorithms/sparkel/run_experiment.py`
**Actual import chain (verified):** `run_experiment.py` → `alzette.py`,
`hybird_attack.py`, `final_hybrid_attack.py`; `final_hybrid_attack.py` also
imports `hybird_attack.py`; `hybird_attack.py` → `mixed_bp_engine.py`,
`byte_bit_factor.py`, `bit_factors.py`. All `numpy`-only.

Attacks the ARX addition (carry-chain) inside Sparkle's Alzette box via a
bit-level BP engine with explicit full-/half-adder factors (`bit_factors.py`).
Config: σ ∈ {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 1.0},
`N_RESTARTS=4, N_ITERS=80, DAMPING=0.4`. Metric is **bits correctly recovered**
out of 64 (A4/A5 halves) or 128 (master key) — not a candidate rank, so no
top-k table applies to this cipher's saved data.

**⚠️ Trial-count note:** `N_TRIALS` in this copy is set to **1000** (originally
50, for cross-cipher consistency). The included `results.json` was **not
regenerated** — it still holds the original **50-trial** run
(`results.json`'s own `config.n_trials` field correctly reports `50`).
Re-running fresh at 1000 trials will take substantially longer (80 iterations
× 4 restarts per trial) and will produce a new `results.json` with tighter
variance than the included 50-trial file.

---

## What's excluded and why

- **`elephant_bit_bp.py`** — an earlier, separate bit-level BP implementation
  considered during drafting. Not imported by `run_experiment.py` or any file
  in this release; not the source of any number in the paper.
- **`luo_recovery/`** (any occurrence) — belongs to a separate paper.
- **`.venv/`, `__pycache__/`, `*.pyc`** — environment/build artifacts.
- **Vendored reference code** (`Xoodoo-master/`, `sparkle-master/`,
  `SCALib-main/`, `isap/`, `ascon/`) — third-party sources; `scalib` is listed
  in `requirements.txt` as a pip dependency instead of vendoring its source.
- **Superseded/toy Xoodyak pipelines** — `codde/attack.py`, `codde/bp_engine.py`,
  and the root-level `xoodyak_bp_*.py` / `xoodyak_rank_table_*.py` scripts.
  Predate the final SCALib-based pipeline, or produced degenerate/toy output
  (e.g. rank always 1); not the source of any number in the paper.
- **An earlier broken SCALib run** — a logged `ImportError` from a prior
  attempt exists in the source repo but is not included; only the final,
  successful `xoodyak_scalib_attack.py` and its successful output are published.
