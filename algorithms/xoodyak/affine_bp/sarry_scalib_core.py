"""
Sarry-style byte-level rank attack on the REAL Xoodyak key, built entirely
on SCALib (FactorGraph + BPState.bp_loopy), like every other Xoodyak
experiment in this folder. Supersedes sarry_byte_rank_core.py, which used
the hand-written Sparkle engine (mixed_bp_engine.py) instead of SCALib.

One graph for the whole first round, NC = 2 (all variables are bits):
  k0..k127      key bits                         VAR SINGLE
  n0..n127      nonce bits    version A: PUB SINGLE (known, exact)
                              version B: VAR SINGLE (unknown)
  one           constant 1                       PUB SINGLE
  ci0..ci383    chi-input bits   PROPERTY ci = XOR(key bits, nonce bits[, one])
                                 terms = nonzero entries of row i of [L_k | L_n],
                                 'one' present iff c0[i] = 1
  na*, t*       chi internals    na = !ci ; t = na & ci
  co0..co383    chi-output bits  co = ci ^ t
  byte-HW leakage: GENERIC factor over the 8 bits of a byte
                   (GenFactor.dense, 2^8 table, exp(-(HW-obs)^2 / 2 sigma^2))
                   on 16 key bytes, 48 chi-input bytes, 48 chi-output bytes,
                   and (version B only) 16 nonce bytes.

Byte convention: byte b of a 128-bit array = bits [8b, 8b+8), LSB first.
A chi byte = 8 consecutive z-bits of one lane of one plane.

Version B never reads the true nonce inside the graph or the evidence
other than through the simulated noisy nonce-byte Hamming weights.
"""
import numpy as np
from scalib.attacks import FactorGraph, BPState, GenFactor

from linear_theta import N_KEY, X, Y, Z, chi_input_index
from linear_theta_CORRECTED_init import (
    build_affine_map, build_nonce_map, key_bits_to_state, state_to_chi_input_bits,
)

N_NONCE = 128
N_CHI = Y * X * Z

L_K, _c0a = build_affine_map(nonce_bytes=bytes(16))
L_N, C0 = build_nonce_map()
assert np.array_equal(_c0a, C0)

COLUMNS = [(chi_input_index(0, x, z), chi_input_index(1, x, z), chi_input_index(2, x, z))
           for x in range(X) for z in range(Z)]
CHI_BYTES = [[chi_input_index(y, x, 8 * p + k) for k in range(8)]
             for y in range(Y) for x in range(X) for p in range(4)]          # 48 groups
KEY_BYTES = [[8 * b + k for k in range(8)] for b in range(16)]

HW8 = np.indices((2,) * 8).sum(axis=0).astype(np.float64)                    # HW of each 8-bit assignment


def chi3(a0, a1, a2):
    return a0 ^ ((1 - a1) & a2), a1 ^ ((1 - a2) & a0), a2 ^ ((1 - a0) & a1)


def graph_text(version):
    lines = ["NC 2", "PUB SINGLE one"]
    lines += [f"VAR SINGLE k{i}" for i in range(N_KEY)]
    kind = "PUB" if version == "A" else "VAR"
    lines += [f"{kind} SINGLE n{i}" for i in range(N_NONCE)]
    for i in range(N_CHI):
        lines += [f"VAR SINGLE ci{i}", f"VAR SINGLE na{i}", f"VAR SINGLE t{i}", f"VAR SINGLE co{i}"]
        terms = [f"k{j}" for j in np.flatnonzero(L_K[i])] + [f"n{j}" for j in np.flatnonzero(L_N[i])]
        if C0[i]:
            terms.append("one")
        lines.append(f"PROPERTY ci{i} = " + " ^ ".join(terms))
        lines.append(f"PROPERTY na{i} = !ci{i}")
    for (i0, i1, i2) in COLUMNS:
        lines += [f"PROPERTY t{i0} = na{i1} & ci{i2}", f"PROPERTY co{i0} = ci{i0} ^ t{i0}",
                  f"PROPERTY t{i1} = na{i2} & ci{i0}", f"PROPERTY co{i1} = ci{i1} ^ t{i1}",
                  f"PROPERTY t{i2} = na{i0} & ci{i1}", f"PROPERTY co{i2} = ci{i2} ^ t{i2}"]
    for b, grp in enumerate(KEY_BYTES):
        lines += [f"GENERIC SINGLE hk{b}", f"PROPERTY HK{b}: hk{b}(" + ",".join(f"k{j}" for j in grp) + ")"]
    if version == "B":
        for b, grp in enumerate(KEY_BYTES):
            lines += [f"GENERIC SINGLE hn{b}", f"PROPERTY HN{b}: hn{b}(" + ",".join(f"n{j}" for j in grp) + ")"]
    for g, grp in enumerate(CHI_BYTES):
        lines += [f"GENERIC SINGLE hi{g}", f"PROPERTY HI{g}: hi{g}(" + ",".join(f"ci{j}" for j in grp) + ")"]
        lines += [f"GENERIC SINGLE ho{g}", f"PROPERTY HO{g}: ho{g}(" + ",".join(f"co{j}" for j in grp) + ")"]
    return "\n".join(lines)


_GRAPHS = {}


def get_graph(version):
    if version not in _GRAPHS:
        _GRAPHS[version] = FactorGraph(graph_text(version))
    return _GRAPHS[version]


def hw_factor(obs, sigma):
    ll = -0.5 * ((HW8 - obs) / sigma) ** 2
    return GenFactor.dense(np.exp(ll - ll.max()))


def simulate(seed, sigma):
    """Key, nonce, true intermediates and all noisy byte-HW observations.
    Separate streams, nonce-byte noise drawn last, so versions A and B see
    identical observations on every leakage point they share."""
    key_bits = np.random.default_rng(seed).integers(0, 2, size=N_KEY).astype(np.uint8)
    nonce_bytes = bytes(np.random.default_rng(seed + 500_000_000).integers(0, 256, 16, dtype=np.uint8))
    nonce_bits = np.unpackbits(np.frombuffer(nonce_bytes, dtype=np.uint8), bitorder="little")
    chi_in = state_to_chi_input_bits(key_bits_to_state(key_bits, nonce_bytes))
    chi_out = np.zeros(N_CHI, dtype=np.uint8)
    for (i0, i1, i2) in COLUMNS:
        chi_out[i0], chi_out[i1], chi_out[i2] = chi3(int(chi_in[i0]), int(chi_in[i1]), int(chi_in[i2]))
    rng = np.random.default_rng(seed + 1_000_000_000)
    obs = {
        "k": [key_bits[g].sum() + rng.normal(0.0, sigma) for g in KEY_BYTES],
        "ci": [chi_in[g].sum() + rng.normal(0.0, sigma) for g in CHI_BYTES],
        "co": [chi_out[g].sum() + rng.normal(0.0, sigma) for g in CHI_BYTES],
    }
    obs["n"] = [nonce_bits[g].sum() + rng.normal(0.0, sigma) for g in KEY_BYTES]
    return key_bits, nonce_bits, chi_in, chi_out, obs


def run_attack(seed, sigma, version, n_iters):
    """Returns (ranks of the 16 true key bytes, 0 = best; nan_flag; max key-bit
    belief change in the last iteration is not exposed by SCALib, so
    convergence is assessed by comparing iteration counts in the pilot)."""
    key_bits, nonce_bits, chi_in, chi_out, obs = simulate(seed, sigma)
    fg = get_graph(version)
    pub = {"one": 1}
    if version == "A":
        pub.update({f"n{i}": int(nonce_bits[i]) for i in range(N_NONCE)})
    gen = {f"hk{b}": hw_factor(obs["k"][b], sigma) for b in range(16)}
    for g in range(48):
        gen[f"hi{g}"] = hw_factor(obs["ci"][g], sigma)
        gen[f"ho{g}"] = hw_factor(obs["co"][g], sigma)
    if version == "B":
        gen.update({f"hn{b}": hw_factor(obs["n"][b], sigma) for b in range(16)})
    bp = BPState(fg, 1, pub, gen)
    bp.bp_loopy(n_iters, initialize_states=True)

    marg = np.array([bp.get_distribution(f"k{i}") for i in range(N_KEY)])     # (128, 2)
    nan_flag = bool(not np.all(np.isfinite(marg)))
    ranks = []
    vals = np.arange(256)
    for b in range(16):
        if nan_flag:
            ranks.append(None)
            continue
        logp = np.zeros(256)
        for k in range(8):
            logp += np.log(np.maximum(marg[8 * b + k][(vals >> k) & 1], 1e-300))
        true_v = int((key_bits[8 * b:8 * b + 8] * (1 << np.arange(8))).sum())
        # rank = number of values strictly more likely + half of ties (fair to ties)
        better = int(np.sum(logp > logp[true_v]))
        ties = int(np.sum(logp == logp[true_v])) - 1
        ranks.append(better + ties / 2.0)
    return ranks, nan_flag


def self_test(n=5):
    """The graph must accept the true assignment for random keys/nonces."""
    for version in ("A", "B"):
        fg = get_graph(version)
        for s in range(n):
            key_bits, nonce_bits, chi_in, chi_out, obs = simulate(777 + s, 1.0)
            na = 1 - chi_in
            t = (chi_out ^ chi_in)
            var = {f"k{i}": int(key_bits[i]) for i in range(N_KEY)}
            pub = {"one": 1}
            (pub if version == "A" else var).update({f"n{i}": int(nonce_bits[i]) for i in range(N_NONCE)})
            for i in range(N_CHI):
                var.update({f"ci{i}": int(chi_in[i]), f"na{i}": int(na[i]), f"t{i}": int(t[i]), f"co{i}": int(chi_out[i])})
            ones = GenFactor.dense(np.ones((2,) * 8))
            gen = {name: ones for name in fg.factors() if False}
            names = [f"hk{b}" for b in range(16)] + [f"hi{g}" for g in range(48)] + [f"ho{g}" for g in range(48)]
            if version == "B":
                names += [f"hn{b}" for b in range(16)]
            fg.sanity_check(pub, var, {nm: ones for nm in names})
    print(f"self_test OK: SCALib graph (versions A and B) accepts the true assignment for {n} random key/nonce pairs")


if __name__ == "__main__":
    self_test()
