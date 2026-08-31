"""
Byte ↔ bit decomposition factor.

Connects one 256-value byte variable to 8 binary bit variables.
Constraint: byte = sum over k=0..7 of (bit_k * 2^k)

This factor lets BP propagate sharp byte-level beliefs into bit-level
representations and vice versa, which is the core of the hybrid approach.
"""

import numpy as np


def make_byte_to_bits_factor(byte_var, bit_vars):
    """
    byte_var: 256-value byte variable name
    bit_vars: list of 8 binary variable names, bit_vars[k] is bit k (LSB first)
    
    Constraint: byte = sum_k bit_vars[k] * 2^k
    """
    assert len(bit_vars) == 8
    
    # Precompute: for each byte value, the 8 bits
    BITS = np.array([[(b >> k) & 1 for k in range(8)] for b in range(256)],
                     dtype=np.int8)  # shape (256, 8)
    
    def msg_fn(others, target):
        if target == byte_var:
            # μ_byte(v) = ∏_k μ_{bit_k}(BITS[v, k])
            # Compute log-prob per byte value, then exp
            out = np.ones(256)
            for k in range(8):
                bk_msg = others[bit_vars[k]]   # shape (2,)
                # Multiply: out[v] *= bk_msg[BITS[v, k]]
                out = out * bk_msg[BITS[:, k]]
            return out
        else:
            # target is one of the bit variables
            try:
                k = bit_vars.index(target)
            except ValueError:
                raise RuntimeError(f"Target {target} not in bit_vars")
            
            mb = others[byte_var]  # shape (256,)
            # μ_{bit_k}(0) = sum over byte values v with bit k = 0: mb[v] * (other bits product)
            # μ_{bit_k}(1) = sum over byte values v with bit k = 1: mb[v] * (other bits product)
            # Other bits product: ∏_{k' ≠ k} μ_{bit_k'}(BITS[v, k'])
            other_prod = np.ones(256)
            for kk in range(8):
                if kk == k:
                    continue
                bkk_msg = others[bit_vars[kk]]
                other_prod = other_prod * bkk_msg[BITS[:, kk]]
            
            # Now sum over byte values, grouped by their bit k
            mask_0 = (BITS[:, k] == 0)
            mask_1 = (BITS[:, k] == 1)
            out = np.zeros(2)
            out[0] = (mb * other_prod * mask_0).sum()
            out[1] = (mb * other_prod * mask_1).sum()
            return out
    
    return msg_fn


# ─────────── Self-test ───────────

def _selftest():
    from mixed_bp_engine import FactorGraph
    
    # Test: byte = 0x4A (= 0b01001010) → bits should match
    fg = FactorGraph()
    fg.add_variable('byte', 256, prior=np.eye(256)[0x4A])
    bit_names = [f'bit{k}' for k in range(8)]
    for bn in bit_names:
        fg.add_variable(bn, 2)
    fg.add_factor('decomp', ['byte'] + bit_names,
                  make_byte_to_bits_factor('byte', bit_names))
    fg.initialize()
    for _ in range(3):
        fg.iterate()
    
    expected = [(0x4A >> k) & 1 for k in range(8)]
    for k in range(8):
        bel = fg.belief(f'bit{k}')
        actual = int(np.argmax(bel))
        assert actual == expected[k], \
            f"bit {k} wrong: got {actual} (probs={bel}), expected {expected[k]}"
    
    # Test reverse: bits set → byte should match
    fg = FactorGraph()
    fg.add_variable('byte', 256)
    # Set bits to spell out 0x73 = 0b01110011
    bit_priors = [(0x73 >> k) & 1 for k in range(8)]
    for k in range(8):
        prior = np.array([1.0, 0.0]) if bit_priors[k] == 0 else np.array([0.0, 1.0])
        fg.add_variable(f'bit{k}', 2, prior=prior)
    fg.add_factor('decomp', ['byte'] + [f'bit{k}' for k in range(8)],
                  make_byte_to_bits_factor('byte', [f'bit{k}' for k in range(8)]))
    fg.initialize()
    for _ in range(3):
        fg.iterate()
    bel = fg.belief('byte')
    assert np.argmax(bel) == 0x73, f"byte wrong: got {np.argmax(bel)}, expected 0x73"
    
    return True


if __name__ == "__main__":
    _selftest()
    print("OK — Byte ↔ bit decomposition factor self-test passes.")
    print("    Forward (byte → bits) and reverse (bits → byte) both work.")