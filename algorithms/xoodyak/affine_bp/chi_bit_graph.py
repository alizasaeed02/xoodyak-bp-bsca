"""
chi_bit_graph.py
=====================================================================
BP/SASCA on chi ONLY, at BIT level (not byte level).

Why bit level, not byte level: byte-Hamming-weight leakage on just the
6 (or even 12, incl. NOT/AND intermediates) byte values of one chi
column is fundamentally ambiguous -- check_ambiguity.py (already in
this project) shows on average 6373 different (ka,kb,kc) byte triples
share the exact same 6-value HW signature, so no inference method can
resolve them from that leakage alone. Piling on more leak points made
loopy BP even less reliable (over-determined cyclic graph -> the
"overconfidence" failure mode flagged in Luo et al. Sec. 3.3 and
Wedenig et al.), not more.

Chi has NO z-mixing (each of the 128 3-bit columns is fully
independent), so leaking each of the 6 physical bits per column
directly (a0,a1,a2 in, b0,b1,b2 out) -- exactly this project's own
published Xoodyak Tanner graph (Fig. 7) -- has no such ambiguity: a
single noisy bit observation determines that bit, not a combinatorial
class of byte values.
=====================================================================
"""
import numpy as np
from scalib.attacks import FactorGraph, BPState

GRAPH_STR = """
NC 2
VAR SINGLE a0
VAR SINGLE a1
VAR SINGLE a2
VAR SINGLE na1
VAR SINGLE na2
VAR SINGLE na0
VAR SINGLE t0
VAR SINGLE t1
VAR SINGLE t2
VAR SINGLE b0
VAR SINGLE b1
VAR SINGLE b2
PROPERTY na1 = !a1
PROPERTY na2 = !a2
PROPERTY na0 = !a0
PROPERTY t0 = na1 & a2
PROPERTY t1 = na2 & a0
PROPERTY t2 = na0 & a1
PROPERTY b0 = a0 ^ t0
PROPERTY b1 = a1 ^ t1
PROPERTY b2 = a2 ^ t2
"""


def build_graph():
    return FactorGraph(GRAPH_STR)


def chi3(a0, a1, a2):
    b0 = a0 ^ ((1 - a1) & a2)
    b1 = a1 ^ ((1 - a2) & a0)
    b2 = a2 ^ ((1 - a0) & a1)
    return b0, b1, b2


def leak_ev(bit_val, sigma, rng, n_traces=1):
    """Per-bit noisy-HW evidence, accumulated in log space over n_traces
    independent noisy observations of the SAME bit (same key, repeated
    measurement) -- averaging reduces effective noise, the standard lever
    used elsewhere in this project (Ascon/Elephant/Sparkle multi-trace)."""
    ll0 = 0.0
    ll1 = 0.0
    for _ in range(n_traces):
        obs = bit_val + rng.normal(0.0, sigma)
        ll0 += -0.5 * (obs - 0.0) ** 2 / sigma ** 2
        ll1 += -0.5 * (obs - 1.0) ** 2 / sigma ** 2
    m = max(ll0, ll1)
    p0, p1 = np.exp(ll0 - m), np.exp(ll1 - m)
    s = p0 + p1
    return np.array([p0 / s, p1 / s])


def attack_column(a0, a1, a2, sigma, rng, fg, n_iters=30, n_traces=1):
    b0, b1, b2 = chi3(a0, a1, a2)
    bp = BPState(fg, 1)
    bp.set_evidence("a0", leak_ev(a0, sigma, rng, n_traces))
    bp.set_evidence("a1", leak_ev(a1, sigma, rng, n_traces))
    bp.set_evidence("a2", leak_ev(a2, sigma, rng, n_traces))
    bp.set_evidence("b0", leak_ev(b0, sigma, rng, n_traces))
    bp.set_evidence("b1", leak_ev(b1, sigma, rng, n_traces))
    bp.set_evidence("b2", leak_ev(b2, sigma, rng, n_traces))
    bp.bp_loopy(n_iters, initialize_states=True)
    a0h = int(np.argmax(bp.get_distribution("a0")))
    a1h = int(np.argmax(bp.get_distribution("a1")))
    a2h = int(np.argmax(bp.get_distribution("a2")))
    return a0h, a1h, a2h


def self_test(n=500, seed=0):
    rng = np.random.default_rng(seed)
    fg = build_graph()
    for _ in range(n):
        a0, a1, a2 = (int(rng.integers(0, 2)) for _ in range(3))
        b0, b1, b2 = chi3(a0, a1, a2)
        bp = BPState(fg, 1)
        bp.set_evidence("a0", np.array([1.0, 0.0]) if a0 == 0 else np.array([0.0, 1.0]))
        bp.set_evidence("a1", np.array([1.0, 0.0]) if a1 == 0 else np.array([0.0, 1.0]))
        bp.set_evidence("a2", np.array([1.0, 0.0]) if a2 == 0 else np.array([0.0, 1.0]))
        bp.bp_loopy(5, initialize_states=True)
        for name, expect in [("b0", b0), ("b1", b1), ("b2", b2)]:
            got = int(np.argmax(bp.get_distribution(name)))
            assert got == expect
    print(f"self_test OK: bit-level chi graph matches chi3() on {n} random triples")


if __name__ == "__main__":
    self_test()
