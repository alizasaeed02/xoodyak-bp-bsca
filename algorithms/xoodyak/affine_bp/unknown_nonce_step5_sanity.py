"""
Unknown-nonce attack, item 5: sanity checks.
  (a) Noiseless round trip: 20 random (key, nonce) pairs, TRUE chi-input
      bits (no noise), basis selected in a RANDOM reliability order.
      Must recover both key AND nonce exactly.
  (b) Negative control: the 128 columns' evidence shuffled (same
      mechanism as the known-nonce audit's shuffled-columns control)
      before BP inference and scoring. Must give ~0% success.
"""
import json
import itertools
import numpy as np

from unknown_nonce_core import (
    M_GLOBAL, C0_GLOBAL, select_basis_in_order, osd_unknown_nonce_decode,
    N_U, N_KEY, N_NONCE, N_CHI, N_COL, COORDS, ALL_HYPS, CLIP, nonce_bytes_to_bits,
    column_coords,
)
from linear_theta_CORRECTED_init import key_bits_to_state, state_to_chi_input_bits
from linear_theta import gf2_inverse
from chi_bit_graph import build_graph, chi3, leak_ev


def noiseless_roundtrip_test(n=20, seed=5555):
    rng = np.random.default_rng(seed)
    ok_count = 0
    for _ in range(n):
        key_bits = rng.integers(0, 2, size=N_KEY).astype(np.uint8)
        nonce_bytes = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
        nonce_bits = nonce_bytes_to_bits(nonce_bytes)
        state = key_bits_to_state(key_bits, nonce_bytes)
        chi_in = state_to_chi_input_bits(state)
        order = rng.permutation(N_CHI)
        basis = select_basis_in_order(M_GLOBAL, order, N_U)
        if len(basis) != N_U:
            continue
        inv_sel = gf2_inverse(M_GLOBAL[basis])
        rhs = chi_in[basis] ^ C0_GLOBAL[basis]
        u_hat = ((inv_sel @ rhs) % 2).astype(np.uint8)
        key_hat, nonce_hat = u_hat[:N_KEY], u_hat[N_KEY:]
        ok_count += int(np.array_equal(key_hat, key_bits) and np.array_equal(nonce_hat, nonce_bits))
    return ok_count, n


def shuffled_columns_control(sigma=0.40, n=30, seed_base=3000):
    from scalib.attacks import BPState
    FG = build_graph()
    rng_shuf = np.random.default_rng(999)

    key_succ0 = key_succ1 = nonce_succ1 = 0
    for k in range(n):
        seed = seed_base + int(sigma * 100) * 1000 + k
        rng_key = np.random.default_rng(seed)
        key_bits = rng_key.integers(0, 2, size=N_KEY).astype(np.uint8)
        rng_nonce = np.random.default_rng(seed + 500_000_000)
        nonce_bytes = bytes(rng_nonce.integers(0, 256, 16, dtype=np.uint8))
        state = key_bits_to_state(key_bits, nonce_bytes)
        chi_in_bits = state_to_chi_input_bits(state)
        rng_noise = np.random.default_rng(seed + 1_000_000_000)

        # compute per-column evidence normally first
        col_evs = []
        for (i0, i1, i2) in COORDS:
            a0, a1, a2 = int(chi_in_bits[i0]), int(chi_in_bits[i1]), int(chi_in_bits[i2])
            b0, b1, b2 = chi3(a0, a1, a2)
            ev = {"a0": leak_ev(a0, sigma, rng_noise), "a1": leak_ev(a1, sigma, rng_noise),
                  "a2": leak_ev(a2, sigma, rng_noise), "b0": leak_ev(b0, sigma, rng_noise),
                  "b1": leak_ev(b1, sigma, rng_noise), "b2": leak_ev(b2, sigma, rng_noise)}
            col_evs.append(ev)

        # shuffle which column's evidence feeds which position
        perm = rng_shuf.permutation(N_COL)
        shuffled = [col_evs[p] for p in perm]

        hard_bp = np.zeros(N_CHI, dtype=np.uint8)
        rel_bp = np.zeros(N_CHI, dtype=np.float64)
        col_log_scores = np.zeros((N_COL, 8), dtype=np.float64)
        for ci, (ev, (i0, i1, i2)) in enumerate(zip(shuffled, COORDS)):
            log_scores = []
            for (h0, h1, h2) in ALL_HYPS:
                bb0, bb1, bb2 = chi3(h0, h1, h2)
                s = (np.log(np.clip(ev["a0"][h0], CLIP, 1.0)) + np.log(np.clip(ev["a1"][h1], CLIP, 1.0)) +
                     np.log(np.clip(ev["a2"][h2], CLIP, 1.0)) + np.log(np.clip(ev["b0"][bb0], CLIP, 1.0)) +
                     np.log(np.clip(ev["b1"][bb1], CLIP, 1.0)) + np.log(np.clip(ev["b2"][bb2], CLIP, 1.0)))
                log_scores.append(s)
            col_log_scores[ci] = np.array(log_scores)

            bp = BPState(FG, 1)
            bp.set_evidence("a0", ev["a0"]); bp.set_evidence("a1", ev["a1"]); bp.set_evidence("a2", ev["a2"])
            bp.set_evidence("b0", ev["b0"]); bp.set_evidence("b1", ev["b1"]); bp.set_evidence("b2", ev["b2"])
            bp.bp_loopy(30, initialize_states=True)
            for bit_i, gi, name in [(0, i0, "a0"), (1, i1, "a1"), (2, i2, "a2")]:
                dist = bp.get_distribution(name)
                hard = int(np.argmax(dist)); hard_bp[gi] = hard
                p_h = max(float(dist[hard]), CLIP); p_o = max(float(dist[1 - hard]), CLIP)
                rel_bp[gi] = abs(np.log(p_h) - np.log(p_o))

        u0, u1, ok = osd_unknown_nonce_decode(hard_bp, rel_bp, col_log_scores)
        key0 = u0[:N_KEY]; key1 = u1[:N_KEY]
        key_succ0 += int(np.array_equal(key0, key_bits))
        key_succ1 += int(np.array_equal(key1, key_bits))

    return key_succ0, key_succ1, n


if __name__ == "__main__":
    ok, n = noiseless_roundtrip_test()
    print(f"(a) Noiseless round trip: {ok}/{n} PASS" if ok == n else f"(a) Noiseless round trip: {ok}/{n} -- FAILURES PRESENT")

    s0, s1, n2 = shuffled_columns_control()
    print(f"(b) Shuffled-columns control (sigma=0.40, n={n2}): order0 key success={s0}/{n2}  order1 key success={s1}/{n2}")

    results = {
        "noiseless_roundtrip": {"ok": ok, "n": n, "pass": ok == n},
        "shuffled_columns_control": {"sigma": 0.40, "n": n2, "order0_key_success": s0, "order1_key_success": s1},
    }
    with open("unknown_nonce_step5_sanity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved unknown_nonce_step5_sanity_results.json")
