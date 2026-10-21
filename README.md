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

## License

MIT
