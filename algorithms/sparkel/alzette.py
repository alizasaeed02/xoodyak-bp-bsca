"""
Alzette ARX-box from the Sparkle suite.

Verified against the official C reference:
https://sparkle-lwc.github.io/sparkle  (also: cryptolu/sparkle on GitHub)

C reference (sparkle.c):
    #define ROT(x, n) (((x) >> (n)) | ((x) << (32-(n))))   // right rotation
    rc = RCON[j>>1];
    state[j]   += ROT(state[j+1], 31);  state[j+1] ^= ROT(state[j], 24);  state[j] ^= rc;
    state[j]   += ROT(state[j+1], 17);  state[j+1] ^= ROT(state[j], 17);  state[j] ^= rc;
    state[j]   += state[j+1];           state[j+1] ^= ROT(state[j], 31);  state[j] ^= rc;
    state[j]   += ROT(state[j+1], 24);  state[j+1] ^= ROT(state[j], 16);  state[j] ^= rc;

Equivalent thesis form (Sarry 2024, eq. 5.7), with `≪` denoting LEFT rotation:
    l'_j = l_{j-1} ⊞ ROL(r_{j-1}, k_j)
    l_j  = l'_j ⊕ α
    r_j  = ROL(l'_j, m_j) ⊕ r_{j-1}

with subround parameters (1-indexed):
    j: 1   2   3   4
    k: 1  15   0   8     (left-rot of r_{j-1} in the addition)
    m: 8  15   1  16     (left-rot of l'_j in the XOR for r_j)

Sparkle round constants (from sparkle.c):
    RCON[0..7] = 0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738,
                 0xBB1185EB, 0x4F7C7B57, 0xCFBFA1C8, 0xC2B3293D

Schwaemm256-128 key loading (from schwaemm.c, Initialize):
    state[0..7]   = nonce   (8 words = 32 bytes, rate part)
    state[8..11]  = key     (4 words = 16 bytes = 128 bits, capacity part)
    sparkle(state, 6, 11)   // STATE_BRANS=6, STEPS_BIG=11

In the very first ARXBOX layer of the very first step:
    Alzette with constant RCON[4] is applied to (state[8],  state[9])  → these are K1
    Alzette with constant RCON[5] is applied to (state[10], state[11]) → these are K2

Therefore the BSCA targets:
    AlzetteA4: input l_0 || r_0 = K1, constant α_4 = RCON[4] = 0xBB1185EB
    AlzetteA5: input l_0 || r_0 = K2, constant α_5 = RCON[5] = 0x4F7C7B57
"""

MASK32 = 0xFFFFFFFF

RCON = [0xB7E15162, 0xBF715880, 0x38B4DA56, 0x324E7738,
        0xBB1185EB, 0x4F7C7B57, 0xCFBFA1C8, 0xC2B3293D]

ALPHA_A4 = RCON[4]
ALPHA_A5 = RCON[5]

# Subround parameters (k_j: left-rot of r in addition; m_j: left-rot of l' in r-update)
ROT_K = [1, 15, 0, 8]
ROT_M = [8, 15, 1, 16]


def ror(x, n):
    n &= 31
    return ((x >> n) | (x << (32 - n))) & MASK32 if n else x & MASK32


def rol(x, n):
    return ror(x, (-n) & 31)


def alzette_subround(l, r, alpha, j):
    """One sub-round j ∈ {0,1,2,3} (0-indexed).
    Returns (l_prime, l_new, r_new)."""
    l_prime = (l + rol(r, ROT_K[j])) & MASK32
    l_new = l_prime ^ alpha
    r_new = rol(l_prime, ROT_M[j]) ^ r
    return l_prime, l_new, r_new


def alzette_full(l0, r0, alpha):
    """Run all 4 Alzette sub-rounds. Returns dict of all intermediates."""
    out = {'l0': l0 & MASK32, 'r0': r0 & MASK32}
    l, r = l0 & MASK32, r0 & MASK32
    for j in range(4):
        lp, l, r = alzette_subround(l, r, alpha, j)
        out[f'lp{j+1}'] = lp
        out[f'l{j+1}'] = l
        out[f'r{j+1}'] = r
    return out


def alzette_reference_c(l, r, alpha):
    """Verbatim port of the C macro chain — used only for cross-checking."""
    for s, t in [(31, 24), (17, 17), (0, 31), (24, 16)]:
        l = (l + ror(r, s)) & MASK32
        r ^= ror(l, t)
        l ^= alpha
    return l, r


# ─────────── Self-verification ───────────

def _self_test(n=2000, seed=0):
    """Verify alzette_full matches alzette_reference_c on n random inputs."""
    import random
    random.seed(seed)
    for _ in range(n):
        l0 = random.randint(0, MASK32)
        r0 = random.randint(0, MASK32)
        for alpha in (ALPHA_A4, ALPHA_A5, RCON[0], RCON[3]):
            v = alzette_full(l0, r0, alpha)
            ref_l, ref_r = alzette_reference_c(l0, r0, alpha)
            assert v['l4'] == ref_l, (
                f"l4 mismatch: l0=0x{l0:08X}, r0=0x{r0:08X}, "
                f"got 0x{v['l4']:08X}, expected 0x{ref_l:08X}")
            assert v['r4'] == ref_r, (
                f"r4 mismatch: l0=0x{l0:08X}, r0=0x{r0:08X}, "
                f"got 0x{v['r4']:08X}, expected 0x{ref_r:08X}")
    return True


if __name__ == "__main__":
    _self_test()
    print(f"OK — Alzette matches the official C reference over 2000 random inputs.")
    print(f"     ALPHA_A4 = 0x{ALPHA_A4:08X}")
    print(f"     ALPHA_A5 = 0x{ALPHA_A5:08X}")
    
    # Show a worked example
    l0, r0 = 0xD9C2825F, 0xA30FEBCF
    v = alzette_full(l0, r0, ALPHA_A4)
    print(f"\nWorked example with K1 = 0x{l0:08X} || 0x{r0:08X}, α = α_4:")
    for name in ('l0','r0','lp1','l1','r1','lp2','l2','r2','lp3','l3','r3','lp4','l4','r4'):
        v_w = v[name]
        bytes_hw = [bin((v_w >> (8*b)) & 0xFF).count('1') for b in range(4)]
        print(f"  {name:>4} = 0x{v_w:08X}   byte-HW = {bytes_hw}")