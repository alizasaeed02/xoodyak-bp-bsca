# Blind Side-Channel Attack on Xoodyak Using Belief Propagation

Code release accompanying the paper "Blind Side-Channel Attack on Xoodyak Using Belief Propagation", including the Elephant and Sparkle reproductions used for cross-cipher comparison.

## Setup

    pip install -r requirements.txt

## Running

Each script is run from its own directory:

    cd algorithms/xoodyak/scalib && python xoodyak_scalib_attack.py
    cd algorithms/elephant       && python run_experiment.py
    cd algorithms/sparkel        && python run_experiment.py

- **Xoodyak**: attacks the χ (chi) step of Xoodyak's Xoodoo permutation using SCALib's belief-propagation engine.
- **Elephant**: attacks the LFSR-based keystream generator using a byte-level BP engine.
- **Sparkle**: attacks the ARX addition inside Sparkle's Alzette box using a bit-level BP engine.

Each script saves its own results (ranks, success flags, timing) into its directory when run.

## What's excluded and why

- `elephant_bit_bp.py` — an earlier, separate bit-level BP implementation considered during drafting. Not used by any script in this release.
- `luo_recovery/` — belongs to a separate paper.
- `.venv/`, `__pycache__/`, `*.pyc` — environment/build artifacts.
- Vendored reference code (`Xoodoo-master/`, `sparkle-master/`, `SCALib-main/`, `isap/`, `ascon/`) — third-party sources; SCALib is listed in `requirements.txt` as a pip dependency instead of being vendored.
- Superseded/toy Xoodyak pipelines (`codde/attack.py`, `codde/bp_engine.py`, root-level `xoodyak_bp_*.py` / `xoodyak_rank_table_*.py`) — predate the final SCALib-based pipeline.

## License

MIT
