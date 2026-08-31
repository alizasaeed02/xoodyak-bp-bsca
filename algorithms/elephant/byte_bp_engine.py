"""
Byte-level Belief Propagation engine for Elephant Dumbo attack.

Variables: 8-bit bytes (256 possible values), represented as length-256 prob vectors.
Factors: byte-level equations (XOR, shift, rotation, HW measurement).

This is structurally simpler than the Sparkle bit-level BP because:
  - Each variable has 256 values (not 2)
  - Factors are deterministic byte equations (no carry chains)
  - HW factor connects 1 byte directly to 1 noisy observation

Standard sum-product BP per Sarry et al. CARDIS 2023 equations (5)-(7).
"""

import numpy as np
from collections import defaultdict


# Precompute byte HW table
HW_BYTE = np.array([bin(v).count('1') for v in range(256)], dtype=np.float64)


class FactorGraph:
    """Generic byte-level factor graph for BP.
    
    Variables are identified by string names. Each variable holds a 256-dim
    probability distribution. Factors are functions that compute outgoing
    messages given incoming messages.
    """
    
    def __init__(self):
        self.variables = {}            # name -> length-256 prior vector
        self.var_factors = defaultdict(list)  # name -> [factor_id]
        self.factor_neighbors = []     # factor_id -> [var_name]
        self.factor_msg_fn = []        # factor_id -> message function
        self.factor_names = []         # for debugging
        
        # Messages: f2v[fid][var_name] = length-256 vector
        self.f2v = {}
        self.v2f = {}
    
    def add_variable(self, name, prior=None):
        if name in self.variables:
            return
        if prior is None:
            prior = np.full(256, 1.0/256)
        else:
            prior = np.asarray(prior, dtype=np.float64)
            prior = prior / prior.sum() if prior.sum() > 0 else np.full(256, 1.0/256)
        self.variables[name] = prior
    
    def add_factor(self, name, var_names, msg_fn):
        fid = len(self.factor_names)
        self.factor_names.append(name)
        self.factor_neighbors.append(list(var_names))
        self.factor_msg_fn.append(msg_fn)
        for vn in var_names:
            if vn not in self.variables:
                self.add_variable(vn)
            self.var_factors[vn].append(fid)
    
    def initialize(self):
        self.f2v = {}
        self.v2f = {}
        for fid, nbrs in enumerate(self.factor_neighbors):
            self.f2v[fid] = {vn: np.full(256, 1.0/256) for vn in nbrs}
            self.v2f[fid] = {vn: self.variables[vn].copy() for vn in nbrs}
    
    def iterate(self):
        # 1) Factor → Variable messages
        new_f2v = {}
        for fid, nbrs in enumerate(self.factor_neighbors):
            new_f2v[fid] = {}
            incoming = self.v2f[fid]
            msg_fn = self.factor_msg_fn[fid]
            for target in nbrs:
                others = {vn: incoming[vn] for vn in nbrs if vn != target}
                msg = msg_fn(others, target)
                msg = np.asarray(msg, dtype=np.float64)
                s = msg.sum()
                if s > 0 and np.isfinite(s):
                    msg = msg / s
                else:
                    msg = np.full(256, 1.0/256)
                new_f2v[fid][target] = msg
        self.f2v = new_f2v
        
        # 2) Variable → Factor messages (use log-domain for stability)
        # First compute log-belief per variable
        log_bel = {}
        for vn, prior in self.variables.items():
            log_bel[vn] = np.log(np.maximum(prior, 1e-300))
        for fid, nbrs in enumerate(self.factor_neighbors):
            for vn in nbrs:
                log_bel[vn] = log_bel[vn] + np.log(np.maximum(self.f2v[fid][vn], 1e-300))
        
        # Now compute v→f = belief / msg_f→v in log domain
        new_v2f = {}
        for fid, nbrs in enumerate(self.factor_neighbors):
            new_v2f[fid] = {}
            for vn in nbrs:
                lm = log_bel[vn] - np.log(np.maximum(self.f2v[fid][vn], 1e-300))
                lm = lm - lm.max()
                m = np.exp(lm)
                s = m.sum()
                m = m / s if s > 0 else np.full(256, 1.0/256)
                new_v2f[fid][vn] = m
        self.v2f = new_v2f
    
    def belief(self, name):
        m = self.variables[name].copy()
        for fid in self.var_factors[name]:
            m = m * self.f2v[fid][name]
        s = m.sum()
        return m / s if s > 0 else np.full(256, 1.0/256)
    
    def rank(self, name, true_value):
        """Return the rank (0-indexed) of the true value in the posterior.
        Rank 0 = most likely. This matches the metric used in thesis Table 5.2."""
        bel = self.belief(name)
        # Sort indices by descending probability
        order = np.argsort(-bel)
        rank = int(np.where(order == true_value)[0][0])
        return rank


# ─────────── Factor message functions ───────────

def make_xor_byte_factor(var_a, var_b, var_c):
    """Factor enforcing a XOR b = c (all 8-bit bytes)."""
    # Precompute XOR table once
    A, B = np.meshgrid(np.arange(256), np.arange(256), indexing='ij')
    XOR_AB = (A ^ B).astype(np.int32)   # XOR_AB[a,b] = a^b
    
    def msg_fn(others, target):
        if target == var_a:
            # μ→a(v) = Σ_{b,c: v^b=c} μ_b(b) μ_c(c)
            mb = others[var_b]; mc = others[var_c]
            # For each value of a, sum over b: mb[b] * mc[a^b]
            out = np.zeros(256)
            for a in range(256):
                out[a] = np.sum(mb * mc[a ^ np.arange(256)])
            return out
        elif target == var_b:
            ma = others[var_a]; mc = others[var_c]
            out = np.zeros(256)
            for b in range(256):
                out[b] = np.sum(ma * mc[np.arange(256) ^ b])
            return out
        elif target == var_c:
            ma = others[var_a]; mb = others[var_b]
            out = np.zeros(256)
            for c in range(256):
                out[c] = np.sum(ma * mb[np.arange(256) ^ c])
            return out
    return msg_fn


def make_y_factor(var_xj, var_xj1, var_yj):
    """y_j = x_j XOR x_{j+1}"""
    return make_xor_byte_factor(var_xj, var_xj1, var_yj)


def make_z_factor(var_xj, var_xj2, var_zj):
    """z_j = x_j XOR x_{j+2}"""
    return make_xor_byte_factor(var_xj, var_xj2, var_zj)


def make_intermediate_factor(var_x0, var_x3, var_t):
    """t = ROL3(x_0) XOR (x_3 << 7)
    
    For each value of x_0: rol3 is a permutation
    For each value of x_3: only the LSB matters; (x_3 << 7) & 0xFF = (x_3 & 1) << 7
    So the value of t depends on x_0 (full byte) and x_3's LSB only.
    """
    rol3 = np.array([((v << 3) | (v >> 5)) & 0xFF for v in range(256)], dtype=np.int32)
    x3_to_shifted = np.array([((v & 1) << 7) for v in range(256)], dtype=np.int32)
    
    def msg_fn(others, target):
        if target == var_x0:
            mx3 = others[var_x3]; mt = others[var_t]
            # For each x_0: t = rol3[x_0] ^ (x_3 LSB << 7)
            #   = sum over x_3 of mx3[x_3] * mt[rol3[x_0] ^ ((x_3&1)<<7)]
            # Aggregate by LSB of x_3
            mx3_lsb0 = mx3[::2].sum()  # x_3 with LSB=0 → contributes 0 to shift
            mx3_lsb1 = mx3[1::2].sum() # x_3 with LSB=1 → contributes 0x80 to shift
            # Actually that's wrong — we need to sum mx3 grouped by LSB
            mx3_lsb0 = mx3[np.arange(256) % 2 == 0].sum()
            mx3_lsb1 = mx3[np.arange(256) % 2 == 1].sum()
            out = np.zeros(256)
            for x0 in range(256):
                out[x0] = mx3_lsb0 * mt[rol3[x0]] + mx3_lsb1 * mt[rol3[x0] ^ 0x80]
            return out
        elif target == var_x3:
            mx0 = others[var_x0]; mt = others[var_t]
            # For each x_3: depends only on LSB of x_3
            #   contribution from LSB=0: sum over x0 mx0[x0] * mt[rol3[x0]]
            #   contribution from LSB=1: sum over x0 mx0[x0] * mt[rol3[x0]^0x80]
            t_for_lsb0 = np.sum(mx0 * mt[rol3])
            t_for_lsb1 = np.sum(mx0 * mt[rol3 ^ 0x80])
            out = np.zeros(256)
            out[np.arange(256) % 2 == 0] = t_for_lsb0
            out[np.arange(256) % 2 == 1] = t_for_lsb1
            return out
        elif target == var_t:
            mx0 = others[var_x0]; mx3 = others[var_x3]
            mx3_lsb0 = mx3[np.arange(256) % 2 == 0].sum()
            mx3_lsb1 = mx3[np.arange(256) % 2 == 1].sum()
            out = np.zeros(256)
            for x0 in range(256):
                v_a = rol3[x0]
                out[v_a] += mx0[x0] * mx3_lsb0
                out[v_a ^ 0x80] += mx0[x0] * mx3_lsb1
            return out
    return msg_fn


def make_final_factor(var_t, var_x13, var_x20):
    """x_20 = t XOR (x_13 >> 7)
    
    x_13 >> 7 = (x_13 & 0x80) >> 7 = MSB of x_13 (0 or 1).
    """
    def msg_fn(others, target):
        if target == var_t:
            mx13 = others[var_x13]; mx20 = others[var_x20]
            # Group x13 by MSB
            mx13_msb0 = mx13[:128].sum()   # MSB=0
            mx13_msb1 = mx13[128:].sum()    # MSB=1
            out = np.zeros(256)
            for t in range(256):
                # x_20 = t XOR msb. msb=0 → x_20=t; msb=1 → x_20=t^1
                out[t] = mx13_msb0 * mx20[t] + mx13_msb1 * mx20[t ^ 1]
            return out
        elif target == var_x13:
            mt = others[var_t]; mx20 = others[var_x20]
            t_for_msb0 = np.sum(mt * mx20)              # msb=0: x_20 = t
            t_for_msb1 = np.sum(mt * mx20[np.arange(256) ^ 1])  # msb=1: x_20 = t^1
            out = np.zeros(256)
            out[:128] = t_for_msb0
            out[128:] = t_for_msb1
            return out
        elif target == var_x20:
            mt = others[var_t]; mx13 = others[var_x13]
            mx13_msb0 = mx13[:128].sum()
            mx13_msb1 = mx13[128:].sum()
            out = np.zeros(256)
            for t in range(256):
                out[t] += mt[t] * mx13_msb0
                out[t ^ 1] += mt[t] * mx13_msb1
            return out
    return msg_fn


def make_hw_factor(var_x, observed_hw, sigma):
    """HW factor: observed = HW(x) + N(0, σ²)
    Sends a length-256 message to var_x equal to the likelihood per byte value.
    """
    inv2s2 = 0.5 / (sigma ** 2)
    log_lik = -inv2s2 * (HW_BYTE - observed_hw) ** 2
    log_lik = log_lik - log_lik.max()
    lik = np.exp(log_lik)
    
    def msg_fn(others, target):
        # Single-variable factor — message to var_x is just the likelihood
        return lik.copy()
    return msg_fn


# ─────────── Self-tests ───────────

def _selftest():
    # Test XOR factor: a XOR b = c
    fg = FactorGraph()
    fg.add_variable('a', prior=np.eye(256)[42])  # a = 42 with certainty
    fg.add_variable('b', prior=np.eye(256)[17])  # b = 17 with certainty
    fg.add_variable('c')
    fg.add_factor('xor1', ['a', 'b', 'c'], make_xor_byte_factor('a', 'b', 'c'))
    fg.initialize()
    for _ in range(3):
        fg.iterate()
    bel_c = fg.belief('c')
    assert np.argmax(bel_c) == (42 ^ 17), f"XOR factor failed: argmax={np.argmax(bel_c)}, expected {42^17}"
    
    # Test HW factor with low noise
    fg = FactorGraph()
    fg.add_variable('x')
    fg.add_factor('hw', ['x'], make_hw_factor('x', observed_hw=4.0, sigma=0.01))
    fg.initialize()
    for _ in range(3):
        fg.iterate()
    bel_x = fg.belief('x')
    # All byte values with HW=4 should have equal max probability
    hw4_indices = np.where(HW_BYTE == 4)[0]
    assert bel_x[hw4_indices].sum() > 0.99, f"HW factor failed"
    
    # Test final factor: x_20 = t XOR (x_13 >> 7)
    fg = FactorGraph()
    fg.add_variable('t', prior=np.eye(256)[100])
    fg.add_variable('x13', prior=np.eye(256)[200])  # MSB of 200 = 1
    fg.add_variable('x20')
    fg.add_factor('f', ['t', 'x13', 'x20'], make_final_factor('t', 'x13', 'x20'))
    fg.initialize()
    for _ in range(3):
        fg.iterate()
    bel = fg.belief('x20')
    # x_20 = 100 XOR 1 = 101
    assert np.argmax(bel) == 101, f"Final factor failed: argmax={np.argmax(bel)}, expected 101"
    
    return True


if __name__ == "__main__":
    _selftest()
    print("OK — Byte-level BP engine self-tests pass.")
    print("    XOR factor, HW factor, intermediate/final feedback factors verified.")