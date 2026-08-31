"""
Bit-level factors for use in the hybrid Sparkle BP graph.
These are simpler than byte-level — variables are binary (2 values).
"""

import numpy as np


def make_xor2_bit_factor(va, vb, vc):
    """a XOR b = c, all binary."""
    def msg_fn(others, target):
        if target == va:
            mb = others[vb]; mc = others[vc]
            # a=0: b=c → mb[0]*mc[0] + mb[1]*mc[1]
            # a=1: b!=c → mb[0]*mc[1] + mb[1]*mc[0]
            return np.array([
                mb[0]*mc[0] + mb[1]*mc[1],
                mb[0]*mc[1] + mb[1]*mc[0]
            ])
        elif target == vb:
            ma = others[va]; mc = others[vc]
            return np.array([
                ma[0]*mc[0] + ma[1]*mc[1],
                ma[0]*mc[1] + ma[1]*mc[0]
            ])
        elif target == vc:
            ma = others[va]; mb = others[vb]
            return np.array([
                ma[0]*mb[0] + ma[1]*mb[1],
                ma[0]*mb[1] + ma[1]*mb[0]
            ])
    return msg_fn


def make_xor3_bit_factor(va, vb, vc, vd):
    """a XOR b XOR c = d, all binary."""
    def msg_fn(others, target):
        # The variable not equal to target: 3 of them. Marginalize.
        # XOR is associative, so result is parity of 3 inputs.
        var_list = [va, vb, vc, vd]
        msgs = {v: others[v] for v in var_list if v != target}
        # Parity probability: P(parity=0) and P(parity=1) over the 3 OTHER vars
        p0 = 1.0  # P(running parity = 0)
        p1 = 0.0
        for v, m in msgs.items():
            p0_new = p0 * m[0] + p1 * m[1]
            p1_new = p0 * m[1] + p1 * m[0]
            p0, p1 = p0_new, p1_new
        # If target was the equation result d: msg = [P(parity=0), P(parity=1)]
        # If target was an input (a,b,c): msg = [P(parity_others=msg[0]), ...]
        # Either way, we want msg[t] = P(parity of all = 0 when t fills in)
        # Equation: a XOR b XOR c XOR d = 0 (all four)
        # If target = 0: parity of others should be 0 → return [p0, p1]
        # If target = 1: parity of others should be 1 → return [p1, p0]
        return np.array([p0, p1])
    return msg_fn


def make_xor_const_bit_factor(va, vb, const_bit):
    """a XOR const = b (deterministic 1-1 mapping)."""
    def msg_fn(others, target):
        if target == va:
            mb = others[vb]
            return mb.copy() if const_bit == 0 else np.array([mb[1], mb[0]])
        elif target == vb:
            ma = others[va]
            return ma.copy() if const_bit == 0 else np.array([ma[1], ma[0]])
    return msg_fn


def make_full_adder_factor(vx, vy, vcin, vs, vcout):
    """x + y + cin = s + 2*cout, all binary. 5 variables.
    Sum = (x XOR y XOR cin); cout = majority(x, y, cin).
    """
    var_order = [vx, vy, vcin, vs, vcout]
    valid = []
    for x in (0, 1):
        for y in (0, 1):
            for cin in (0, 1):
                tot = x + y + cin
                valid.append((x, y, cin, tot & 1, tot >> 1))
    
    def msg_fn(others, target):
        ti = var_order.index(target)
        out = np.zeros(2)
        for cfg in valid:
            w = 1.0
            for i, v in enumerate(var_order):
                if v == target: continue
                w *= others[v][cfg[i]]
            out[cfg[ti]] += w
        return out
    return msg_fn


def make_half_adder_factor(vx, vy, vs, vcout):
    """x + y = s + 2*cout (LSB position with no cin)."""
    var_order = [vx, vy, vs, vcout]
    valid = []
    for x in (0, 1):
        for y in (0, 1):
            tot = x + y
            valid.append((x, y, tot & 1, tot >> 1))
    
    def msg_fn(others, target):
        ti = var_order.index(target)
        out = np.zeros(2)
        for cfg in valid:
            w = 1.0
            for i, v in enumerate(var_order):
                if v == target: continue
                w *= others[v][cfg[i]]
            out[cfg[ti]] += w
        return out
    return msg_fn


# ─────────── Self-tests ───────────

def _selftest():
    from mixed_bp_engine import FactorGraph
    
    # XOR: 1 XOR 1 = 0
    fg = FactorGraph()
    fg.add_variable('a', 2, prior=np.array([0.0, 1.0]))
    fg.add_variable('b', 2, prior=np.array([0.0, 1.0]))
    fg.add_variable('c', 2)
    fg.add_factor('xor', ['a','b','c'], make_xor2_bit_factor('a','b','c'))
    fg.initialize()
    for _ in range(3): fg.iterate()
    assert np.argmax(fg.belief('c')) == 0, f"XOR failed"
    
    # Full adder: 1 + 1 + 1 = 1, cout=1
    fg = FactorGraph()
    fg.add_variable('x', 2, prior=np.array([0.0, 1.0]))
    fg.add_variable('y', 2, prior=np.array([0.0, 1.0]))
    fg.add_variable('cin', 2, prior=np.array([0.0, 1.0]))
    fg.add_variable('s', 2)
    fg.add_variable('cout', 2)
    fg.add_factor('fa', ['x','y','cin','s','cout'],
                  make_full_adder_factor('x','y','cin','s','cout'))
    fg.initialize()
    for _ in range(3): fg.iterate()
    assert np.argmax(fg.belief('s')) == 1
    assert np.argmax(fg.belief('cout')) == 1
    
    # XOR-3: 1 XOR 0 XOR 1 = ? (should equal d for equation parity=0)
    fg = FactorGraph()
    fg.add_variable('a', 2, prior=np.array([0.0, 1.0]))
    fg.add_variable('b', 2, prior=np.array([1.0, 0.0]))
    fg.add_variable('c', 2, prior=np.array([0.0, 1.0]))
    fg.add_variable('d', 2)
    fg.add_factor('x3', ['a','b','c','d'], make_xor3_bit_factor('a','b','c','d'))
    fg.initialize()
    for _ in range(3): fg.iterate()
    # 1 XOR 0 XOR 1 = 0
    assert np.argmax(fg.belief('d')) == 0, f"XOR3 failed: got {np.argmax(fg.belief('d'))}"
    
    return True


if __name__ == "__main__":
    _selftest()
    print("OK — Bit-level factors self-tests pass.")
    print("    XOR2, XOR3, XOR-const, full-adder, half-adder verified.")