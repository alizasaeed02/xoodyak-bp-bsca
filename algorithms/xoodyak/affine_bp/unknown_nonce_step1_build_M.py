"""
Unknown-nonce attack, item 1: build the combined key+nonce affine map
M = [L_k | L_n] (384x256) and c0 (chi-input bits at key=0, nonce=0 --
contains only the fixed Cyclist bytes 0x10, 0x01, 0x02 and the round
constant, no key or nonce dependence). Verify c = M @ u XOR c0 exactly
for 50 random (key, nonce) pairs, u = (key_bits, nonce_bits) concatenated
in that order (first 128 columns of M = L_k, next 128 = L_n).

Does not touch any existing file; new module only.
"""
import numpy as np

from linear_theta_CORRECTED_init import (
    build_affine_map, build_nonce_map, key_bits_to_state, state_to_chi_input_bits,
)
from linear_theta import N_KEY

N_NONCE = 128
N_U = N_KEY + N_NONCE  # 256


def build_M_c0():
    L_k, c0_a = build_affine_map(nonce_bytes=bytes(16))  # nonce=0 baseline for L_k's own differencing (nonce-independent, proven earlier)
    L_n, c0_b = build_nonce_map()  # key=0 baseline
    assert np.array_equal(c0_a, c0_b), "c0 mismatch between the two baselines"
    M = np.concatenate([L_k, L_n], axis=1)  # (384, 256)
    return M, c0_a


def nonce_bytes_to_bits(nonce_bytes):
    """Same bit convention as linear_theta_CORRECTED_init.build_nonce_map's
    differential basis: bit i (0..127) = byte i//8, bit i%8 (numpy unpackbits,
    bitorder='little')."""
    return np.unpackbits(np.frombuffer(bytes(nonce_bytes), dtype=np.uint8), bitorder="little")


if __name__ == "__main__":
    M, c0 = build_M_c0()
    print("M shape:", M.shape, " c0 shape:", c0.shape)
    print("c0 nonzero bit positions:", np.flatnonzero(c0).tolist())

    rng = np.random.default_rng(2024)
    all_ok = True
    for i in range(50):
        key_bits = rng.integers(0, 2, size=N_KEY).astype(np.uint8)
        nonce_bytes = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
        nonce_bits = nonce_bytes_to_bits(nonce_bytes)
        u = np.concatenate([key_bits, nonce_bits])

        state = key_bits_to_state(key_bits, nonce_bytes)
        chi_in = state_to_chi_input_bits(state)

        pred = ((M @ u) % 2) ^ c0
        ok = np.array_equal(pred.astype(np.uint8), chi_in)
        all_ok &= ok
        if not ok:
            print(f"  MISMATCH at pair {i}")

    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'} -- c = M@u XOR c0 verified exactly "
          f"for 50 random (key, nonce) pairs.")

    # rank, info only (already established earlier in the audit, re-confirmed here)
    from linear_theta_CORRECTED_init import gf2_rank
    rank = gf2_rank(M)
    print(f"rank(M) = {rank} (full column rank would be {N_U})")
