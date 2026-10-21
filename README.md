# Blind Side-Channel Attack on Xoodyak Using Belief Propagation

Code for the paper *"Blind Side-Channel Attack on Xoodyak Using Belief Propagation and
SASCA"*.

This repository implements a blind side-channel attack (BSCA) on Xoodyak: it models the
nonlinear χ step of the Xoodoo permutation as a factor graph and recovers the secret key
from noisy Hamming-weight leakage using belief propagation (BP). Before attacking Xoodyak,
the same belief-propagation machinery is validated by reproducing three published attacks:

- **Elephant** and **Sparkle** — BP-based blind side-channel attacks, reproduced from
  Sarry et al.
- **Keccak** — a single-trace attack, reproduced using the original authors'
  (Kannwischer, Pessl and Primas) own unmodified code.

That Keccak reproduction is what led to the paper's main finding: naively copying
Keccak's whole-round BP design onto Xoodyak's key-load is actually wrong, because
Xoodyak's linear diffusion layer becomes a fixed, exactly invertible map that should be
solved with linear algebra, not belief propagation. The Xoodyak attack here uses the
corrected design and recovers the full 128-bit key, both from a single trace and from
multiple repeated traces.

## Setup

```
pip install -r requirements.txt
```

## Running

```
cd algorithms/xoodyak/scalib && python xoodyak_scalib_attack.py
cd algorithms/elephant       && python run_experiment.py
cd algorithms/sparkel        && python run_experiment.py
```

Each script saves its own results (ranks, success rates, timing) into its directory when
run.

## Elephant

`algorithms/elephant/` reproduces the BP-based blind attack of Sarry et al. on Elephant
(Dumbo). Ranks are counted from 0 (rank 0 = most likely value; random guessing = 127.5).
The saved `results.json` uses 100 random keys per noise level (`config.n_trials = 100`).
Note that `run_experiment.py` is currently set to `N_TRIALS = 1000`, so running it as is
uses 1000 keys per noise level, not the 100 behind `results.json`.

## Sparkle

`algorithms/sparkel/` reproduces the BP-based blind attack of Sarry et al. on Sparkle
(Alzette). The saved `results.json` uses 50 random keys per noise level
(`config.n_trials = 50`).

## Xoodyak byte-level rank table (journal Table)

The byte-level rank table in the journal paper (for example mean expected rank 14.71 at
σ = 0.10 with the nonce known) comes from `algorithms/xoodyak/affine_bp/`.

```
cd algorithms/xoodyak/affine_bp
python sarry_scalib_final_checks.py ties     # rerun all rows, store tie counts (about 20 min, 4 workers)
python sarry_scalib_final_checks.py report   # build the table and the NaN accounting
python sarry_scalib_final_checks.py sat      # optional: SAT uniqueness check (needs python-sat; slow)
```

The script needs one of the step names above; it does not run without one.

Settings: SCALib 0.6.4 loopy BP on the real first round of Xoodyak (`sarry_scalib_core.py`),
simulated byte Hamming-weight leakage, 1000 keys per row, 10 BP iterations, joint key-byte
read-out, versions A (nonce known) and B (nonce unknown), σ = 0.01 (near-noiseless),
0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00. The rank is the expected rank counted from 1
with ties broken at random (rank = better + (ties + 1) / 2); random guessing = 128.5.

Output tables: `algorithms/xoodyak/affine_bp/sarry_scalib_joint_grid_table_v2.md`,
`.csv` and `.tex`. Per-trial ranks and tie counts are in
`sarry_scalib_joint_grid_ties_trials.jsonl`, and the summary and NaN accounting are in
`sarry_scalib_final_checks_report.json` and `.log`. The `report` step also reads
`sarry_scalib_joint_grid_trials.jsonl`, `sarry_scalib_joint_grid_summary.json` and
`sarry_scalib_diag2_step3_trials.jsonl`, which are included.

The older script `algorithms/xoodyak/scalib/xoodyak_scalib_attack.py` uses a different
rank convention (counted from 0, 20 BP iterations) and is not the source of the journal
table.

## License

MIT
