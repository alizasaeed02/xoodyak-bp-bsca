"""
Hybrid byte/bit BSCA on Alzette.

Architecture:
  - Byte variables for HW evidence (sharp 256-dim posteriors)
  - Byte XOR-α factor (clean, byte-aligned)
  - Byte ↔ bit decomposition factors at addition boundaries
  - Bit variables + full-adder chain for modular addition
  - Bit-level XOR factors for cross-byte rotations

Goal: see if sharper byte-level evidence helps BP escape the 33/64 plateau.
"""

import numpy as np
from numpy.random import default_rng

from alzette import alzette_full, ALPHA_A4, ALPHA_A5, ROT_K, ROT_M, MASK32, rol
from mixed_bp_engine import FactorGraph, HW_BYTE, make_hw_byte_factor, make_xor_const_byte_factor
from byte_bit_factor import make_byte_to_bits_factor
from bit_factors import (
    make_xor2_bit_factor, make_xor_const_bit_factor,
    make_full_adder_factor, make_half_adder_factor,
)


# ─────────── Helpers ───────────

def add_byte_with_bits(fg, byte_name, bit_prefix):
    """Add a byte var with cardinality 256, plus 8 bit vars, plus the
    decomposition factor linking them. Returns the list of 8 bit names."""
    fg.add_variable(byte_name, 256)
    bit_names = [f'{bit_prefix}_{j}' for j in range(8)]
    for bn in bit_names:
        fg.add_variable(bn, 2)
    fg.add_factor(f'decomp_{byte_name}', [byte_name] + bit_names,
                  make_byte_to_bits_factor(byte_name, bit_names))
    return bit_names


def add_pure_bits(fg, prefix, n=32):
    """Add n bit variables. Returns names list."""
    names = [f'{prefix}_{j}' for j in range(n)]
    for nm in names:
        fg.add_variable(nm, 2)
    return names


# ─────────── Build the attack graph ───────────

def build_alzette_graph(obs, alpha, sigma):
    """
    obs: dict with keys 'lp1','l1','r1','lp2','l2','r2','lp3','l3','r3','lp4','l4','r4'
         each is a length-4 array of noisy byte HWs
    alpha: 32-bit constant (ALPHA_A4 or ALPHA_A5)
    sigma: noise level
    
    Returns: (fg, l0_bit_names, r0_bit_names) — graph plus the names of the
    32 bit variables for l_0 and r_0 (the secret).
    """
    fg = FactorGraph()
    
    # ── For each variable l_i, l'_i, r_i, create 32 bit vars ──
    # l_i (i=0..4): 32 bits each
    # l'_i (i=1..4): 32 bits each
    # r_i (i=0..4): 32 bits each
    
    # 32-bit registers split as 4 bytes; each byte has 8 bits.
    # bit j is in byte (j // 8), bit-within-byte (j % 8).
    
    l_bits = {}    # l_bits[i] = list of 32 bit names
    lp_bits = {}
    r_bits = {}
    
    # For the LEAKED words, we additionally have byte vars + decomposition
    l_bytes = {}   # l_bytes[i] = list of 4 byte names (for i=1..4)
    lp_bytes = {}
    r_bytes = {}
    
    # l_0, r_0: pure bits (no leakage on these)
    l_bits[0] = add_pure_bits(fg, 'l0', n=32)
    r_bits[0] = add_pure_bits(fg, 'r0', n=32)
    
    # For i=1..4: byte vars (with HW factors), decomposed into bits
    for i in range(1, 5):
        l_bytes[i] = []
        l_bits[i] = []
        lp_bytes[i] = []
        lp_bits[i] = []
        r_bytes[i] = []
        r_bits[i] = []
        
        for b in range(4):
            # l_i byte b
            ln = f'l{i}_byte{b}'
            fg.add_variable(ln, 256)
            l_bytes[i].append(ln)
            bit_names = add_pure_bits(fg, f'l{i}_b{b}_bit', n=8)
            l_bits[i].extend(bit_names)
            fg.add_factor(f'decomp_l{i}_b{b}', [ln] + bit_names,
                          make_byte_to_bits_factor(ln, bit_names))
            # HW factor
            fg.add_factor(f'HW_l{i}_b{b}', [ln],
                          make_hw_byte_factor(ln, obs[f'l{i}'][b], sigma))
            
            # lp_i byte b
            lpn = f'lp{i}_byte{b}'
            fg.add_variable(lpn, 256)
            lp_bytes[i].append(lpn)
            lp_bit_names = add_pure_bits(fg, f'lp{i}_b{b}_bit', n=8)
            lp_bits[i].extend(lp_bit_names)
            fg.add_factor(f'decomp_lp{i}_b{b}', [lpn] + lp_bit_names,
                          make_byte_to_bits_factor(lpn, lp_bit_names))
            fg.add_factor(f'HW_lp{i}_b{b}', [lpn],
                          make_hw_byte_factor(lpn, obs[f'lp{i}'][b], sigma))
            
            # r_i byte b
            rn = f'r{i}_byte{b}'
            fg.add_variable(rn, 256)
            r_bytes[i].append(rn)
            r_bit_names = add_pure_bits(fg, f'r{i}_b{b}_bit', n=8)
            r_bits[i].extend(r_bit_names)
            fg.add_factor(f'decomp_r{i}_b{b}', [rn] + r_bit_names,
                          make_byte_to_bits_factor(rn, r_bit_names))
            fg.add_factor(f'HW_r{i}_b{b}', [rn],
                          make_hw_byte_factor(rn, obs[f'r{i}'][b], sigma))
    
    # ── XOR-α: l_i = lp_i XOR α (BYTE-LEVEL, since α has byte-aligned bytes) ──
    alpha_bytes = [(alpha >> (8 * b)) & 0xFF for b in range(4)]
    for i in range(1, 5):
        for b in range(4):
            fg.add_factor(f'XORalpha_l{i}_b{b}',
                          [lp_bytes[i][b], l_bytes[i][b]],
                          make_xor_const_byte_factor(lp_bytes[i][b], l_bytes[i][b],
                                                      alpha_bytes[b]))
    
    # ── Modular addition: lp_i = l_{i-1} + (r_{i-1} ROL k_i) ──
    # All bit-level. Carry chain has 31 carries per round.
    for i in range(1, 5):
        k = ROT_K[i - 1]
        # rotated_r_bits[j] = r_bits[i-1][(j - k) mod 32]
        rotated_r_bits = [r_bits[i-1][(j - k) % 32] for j in range(32)]
        
        # Add carries
        carry_names = [f'c{i}_{j}' for j in range(31)]
        for cn in carry_names:
            fg.add_variable(cn, 2)
        
        # Bit 0: half-adder (no cin)
        fg.add_factor(f'HA_lp{i}_0',
                      [l_bits[i-1][0], rotated_r_bits[0], lp_bits[i][0], carry_names[0]],
                      make_half_adder_factor(l_bits[i-1][0], rotated_r_bits[0],
                                              lp_bits[i][0], carry_names[0]))
        # Bits 1..30: full-adders
        for j in range(1, 31):
            fg.add_factor(f'FA_lp{i}_{j}',
                          [l_bits[i-1][j], rotated_r_bits[j], carry_names[j-1],
                           lp_bits[i][j], carry_names[j]],
                          make_full_adder_factor(l_bits[i-1][j], rotated_r_bits[j],
                                                  carry_names[j-1], lp_bits[i][j],
                                                  carry_names[j]))
        # Bit 31: half-adder-style (no cout, since MSB carry is dropped in mod 2^32)
        # Actually it's 3-input XOR (no cout): l+r+cin mod 2 = s, cout discarded
        # We use a fake cout that's free, so it's a regular full-adder with discarded cout
        fake_cout = f'fake_cout_lp{i}_31'
        fg.add_variable(fake_cout, 2)
        fg.add_factor(f'FA_lp{i}_31',
                      [l_bits[i-1][31], rotated_r_bits[31], carry_names[30],
                       lp_bits[i][31], fake_cout],
                      make_full_adder_factor(l_bits[i-1][31], rotated_r_bits[31],
                                              carry_names[30], lp_bits[i][31],
                                              fake_cout))
    
    # ── XOR for r update: r_i = (lp_i ROL m_i) XOR r_{i-1} (bit-level) ──
    for i in range(1, 5):
        m = ROT_M[i - 1]
        for j in range(32):
            # r_i_bits[j] = lp_i_bits[(j - m) mod 32] XOR r_{i-1}_bits[j]
            fg.add_factor(f'XORr_{i}_{j}',
                          [lp_bits[i][(j - m) % 32], r_bits[i-1][j], r_bits[i][j]],
                          make_xor2_bit_factor(lp_bits[i][(j - m) % 32],
                                                r_bits[i-1][j],
                                                r_bits[i][j]))
    
    return fg, l_bits[0], r_bits[0]


# ─────────── Run the attack ───────────

def attack_one(l0, r0, alpha, sigma, rng, n_iters=30, verbose=False):
    """Run hybrid BP attack on a single Alzette."""
    # Simulate leakage
    vals = alzette_full(l0, r0, alpha)
    obs = {}
    for w in ('lp1','l1','r1','lp2','l2','r2','lp3','l3','r3','lp4','l4','r4'):
        x = vals[w]
        h = np.array([bin((x >> (8*b)) & 0xFF).count('1') for b in range(4)],
                     dtype=float)
        h += rng.normal(0, sigma, 4)
        obs[w] = h
    
    # Build graph
    fg, l0_bits, r0_bits = build_alzette_graph(obs, alpha, sigma)
    fg.initialize()
    
    # Iterate
    for it in range(n_iters):
        fg.iterate()
        if verbose and (it < 5 or it % 5 == 0):
            l0_hat = sum((1 if fg.belief(l0_bits[j])[1] > 0.5 else 0) << j for j in range(32))
            r0_hat = sum((1 if fg.belief(r0_bits[j])[1] > 0.5 else 0) << j for j in range(32))
            cl0 = 32 - bin(l0_hat ^ l0).count('1')
            cr0 = 32 - bin(r0_hat ^ r0).count('1')
            print(f"  iter {it+1:2d}: l0={cl0:2d}  r0={cr0:2d}  total={cl0+cr0:2d}/64")
    
    # Final hard decisions
    l0_hat = sum((1 if fg.belief(l0_bits[j])[1] > 0.5 else 0) << j for j in range(32))
    r0_hat = sum((1 if fg.belief(r0_bits[j])[1] > 0.5 else 0) << j for j in range(32))
    cl0 = 32 - bin(l0_hat ^ l0).count('1')
    cr0 = 32 - bin(r0_hat ^ r0).count('1')
    return cl0 + cr0, cl0, cr0


if __name__ == "__main__":
    import time
    rng = default_rng(0)
    print("Hybrid byte/bit Sparkle BSCA — demo")
    print("=" * 60)
    print("3 trials at σ=0.1, AlzetteA4")
    print()
    for trial in range(3):
        l0 = int(rng.integers(0, 2**32))
        r0 = int(rng.integers(0, 2**32))
        t0 = time.time()
        total, cl0, cr0 = attack_one(l0, r0, ALPHA_A4, 0.1, rng,
                                      n_iters=30,
                                      verbose=(trial == 0))
        elapsed = time.time() - t0
        print(f"trial {trial+1}: l0={cl0}/32  r0={cr0}/32  TOTAL = {total}/64  ({elapsed:.1f}s)")
    print()
    print("Compare: paper Table 5.3 reports mean=57/64 at σ=0.1")
    print("Original bit-level BP plateau: ~33/64")