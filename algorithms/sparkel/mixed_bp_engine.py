"""
Mixed-arity belief propagation engine.

Each variable has its own cardinality (e.g. 2 for bits, 256 for bytes).
Messages are length-cardinality probability vectors.

Standard sum-product BP, log-domain v→f messages for stability.
"""

import numpy as np
from collections import defaultdict


# Precompute byte HW table
HW_BYTE = np.array([bin(v).count('1') for v in range(256)], dtype=np.float64)


class FactorGraph:
    def __init__(self):
        self.var_card = {}                           # name -> cardinality
        self.variables = {}                          # name -> length-cardinality prior
        self.var_factors = defaultdict(list)         # name -> [factor_id]
        self.factor_neighbors = []                   # factor_id -> [var_name]
        self.factor_msg_fn = []
        self.factor_names = []
        
        self.f2v = {}
        self.v2f = {}
    
    def add_variable(self, name, cardinality, prior=None):
        if name in self.var_card:
            assert self.var_card[name] == cardinality, f"Variable {name} cardinality mismatch"
            return
        self.var_card[name] = cardinality
        if prior is None:
            prior = np.full(cardinality, 1.0 / cardinality)
        else:
            prior = np.asarray(prior, dtype=np.float64)
            s = prior.sum()
            prior = prior / s if s > 0 else np.full(cardinality, 1.0 / cardinality)
        self.variables[name] = prior
    
    def add_factor(self, name, var_names, msg_fn):
        fid = len(self.factor_names)
        self.factor_names.append(name)
        self.factor_neighbors.append(list(var_names))
        self.factor_msg_fn.append(msg_fn)
        for vn in var_names:
            assert vn in self.var_card, f"Variable {vn} must be added before factor {name}"
            self.var_factors[vn].append(fid)
    
    def initialize(self):
        self.f2v = {}
        self.v2f = {}
        for fid, nbrs in enumerate(self.factor_neighbors):
            self.f2v[fid] = {vn: np.full(self.var_card[vn], 1.0 / self.var_card[vn])
                             for vn in nbrs}
            self.v2f[fid] = {vn: self.variables[vn].copy() for vn in nbrs}
    
    def iterate(self):
        # 1) Factor → Variable
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
                    card = self.var_card[target]
                    msg = np.full(card, 1.0 / card)
                new_f2v[fid][target] = msg
        self.f2v = new_f2v
        
        # 2) Variable → Factor (log-domain)
        log_bel = {vn: np.log(np.maximum(self.variables[vn], 1e-300))
                   for vn in self.var_card}
        for fid, nbrs in enumerate(self.factor_neighbors):
            for vn in nbrs:
                log_bel[vn] = log_bel[vn] + np.log(np.maximum(self.f2v[fid][vn], 1e-300))
        
        new_v2f = {}
        for fid, nbrs in enumerate(self.factor_neighbors):
            new_v2f[fid] = {}
            for vn in nbrs:
                lm = log_bel[vn] - np.log(np.maximum(self.f2v[fid][vn], 1e-300))
                lm = lm - lm.max()
                m = np.exp(lm)
                s = m.sum()
                card = self.var_card[vn]
                m = m / s if s > 0 else np.full(card, 1.0 / card)
                new_v2f[fid][vn] = m
        self.v2f = new_v2f
    
    def belief(self, name):
        m = self.variables[name].copy()
        for fid in self.var_factors[name]:
            m = m * self.f2v[fid][name]
        s = m.sum()
        card = self.var_card[name]
        return m / s if s > 0 else np.full(card, 1.0 / card)


# ─────────── Generic factor builders ───────────

def make_hw_byte_factor(var_x, observed_hw, sigma):
    """HW factor on a byte variable (256 values)."""
    inv2s2 = 0.5 / (sigma ** 2)
    log_lik = -inv2s2 * (HW_BYTE - observed_hw) ** 2
    log_lik = log_lik - log_lik.max()
    lik = np.exp(log_lik)
    
    def msg_fn(others, target):
        return lik.copy()
    return msg_fn


def make_xor_byte_factor(var_a, var_b, var_c):
    """Factor enforcing a XOR b = c (all 8-bit bytes)."""
    def msg_fn(others, target):
        if target == var_a:
            mb = others[var_b]; mc = others[var_c]
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


def make_xor_const_byte_factor(var_a, var_b, const_byte):
    """Factor: a XOR const = b. Bijective so easy."""
    def msg_fn(others, target):
        if target == var_a:
            mb = others[var_b]
            return mb[np.arange(256) ^ const_byte]
        elif target == var_b:
            ma = others[var_a]
            return ma[np.arange(256) ^ const_byte]
    return msg_fn


# ─────────── Self-tests ───────────

def _selftest():
    fg = FactorGraph()
    fg.add_variable('a', 256, prior=np.eye(256)[42])
    fg.add_variable('b', 256, prior=np.eye(256)[17])
    fg.add_variable('c', 256)
    fg.add_factor('xor', ['a','b','c'], make_xor_byte_factor('a','b','c'))
    fg.initialize()
    for _ in range(3): fg.iterate()
    assert np.argmax(fg.belief('c')) == (42 ^ 17), f"XOR failed"
    
    fg = FactorGraph()
    fg.add_variable('x', 256)
    fg.add_factor('hw', ['x'], make_hw_byte_factor('x', 4.0, 0.01))
    fg.initialize()
    for _ in range(3): fg.iterate()
    bel = fg.belief('x')
    hw4 = np.where(HW_BYTE == 4)[0]
    assert bel[hw4].sum() > 0.99, f"HW failed"
    
    # Mixed cardinality test
    fg = FactorGraph()
    fg.add_variable('byte_var', 256, prior=np.eye(256)[100])
    fg.add_variable('bit_var', 2, prior=np.array([1.0, 0.0]))
    fg.add_variable('result', 256)
    
    def custom_factor(others, target):
        if target == 'byte_var':
            mbit = others['bit_var']; mres = others['result']
            out = np.zeros(256)
            for b in range(256):
                out[b] = mbit[0] * mres[b] + mbit[1] * mres[b ^ 1]
            return out
        elif target == 'bit_var':
            mb = others['byte_var']; mres = others['result']
            out = np.zeros(2)
            out[0] = np.sum(mb * mres)
            out[1] = np.sum(mb * mres[np.arange(256) ^ 1])
            return out
        elif target == 'result':
            mb = others['byte_var']; mbit = others['bit_var']
            out = np.zeros(256)
            for b in range(256):
                out[b] += mb[b] * mbit[0]
                out[b ^ 1] += mb[b] * mbit[1]
            return out
    
    fg.add_factor('mixed', ['byte_var', 'bit_var', 'result'], custom_factor)
    fg.initialize()
    for _ in range(3): fg.iterate()
    # Expect result = 100 XOR 0 = 100
    assert np.argmax(fg.belief('result')) == 100, f"Mixed failed: got {np.argmax(fg.belief('result'))}"
    
    return True


if __name__ == "__main__":
    _selftest()
    print("OK — Mixed-arity BP engine self-tests pass.")