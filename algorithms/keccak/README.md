# Keccak reproduction

Reproduces Kannwischer, Pessl and Primas, *"Single-Trace Attacks on Keccak"*
(TCHES 2020, Issue 3), using their own public, unmodified code
(https://github.com/keccaksasca/keccaksasca), at their stated experimental
protocol: 100 trials per noise level, 8-bit device, 128-bit secret,
Keccak-f[1600], 2 rounds.

The only local change to their code is `shim/ffht.py`, a pure-Python
drop-in for the `ffht` C extension, needed because the compiled extension
is unavailable / ABI-incompatible in some environments. It is a plain
Walsh-Hadamard transform and does not change any attack logic.

## Which script produced which number

| Script | Scenario | sigma values | Output |
|---|---|---|---|
| `run_keccak_random_input.py` | random-but-known public input | 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0 | `results.json["random_input"]` |
| `run_keccak_allzero_input.py` | all-zero public input | 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0 | `results.json["all_zero_input"]` |

Both scripts run the authors' `bp.py` as a subprocess per (sigma, seed) pair
and parse its own `LOG ...` output line; success = `numerrors_in == 0` in
that line (the authors' own success criterion). `n_valid` in the results may
be below 100 where an individual run hit the 900s timeout (all such cases
are recorded, not silently dropped).

## Files

- `run_keccak_random_input.py`, `run_keccak_allzero_input.py` — driver scripts
- `shim/ffht.py` — pure-Python fallback for the `ffht` C extension
- `results.json` — merged output of both scripts (real executed data, n=100/point)
- `keccak_reproduction.png` — success-rate-vs-noise plot, laid out to sit next
  to the original paper's Figure 4 for direct comparison

## Reproducing

```
git clone https://github.com/keccaksasca/keccaksasca.git
# place keccaksasca/keccak-p next to these scripts, or edit REPO in each script
python run_keccak_random_input.py
python run_keccak_allzero_input.py
```
