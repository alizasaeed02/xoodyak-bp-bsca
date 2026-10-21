"""
Diagnostic for the byte-level rank table: genuine leakage/inference limit, or a
read-out / implementation problem?   BP: SCALib 0.6.4, graph = sarry_scalib_core.py
(unmodified). No manuscript edits, no full grid.

  step1  exact-HW test (indicator likelihoods), 20 key/nonce pairs, A and B:
         (a) true assignment satisfies every constraint, likelihood non-zero;
         (b) how many key-byte values / whole keys remain consistent with the exact
             observations. Counting is done with a SAT solver (python-sat, CaDiCaL)
             on the same constraints -- a counting tool only, not an inference engine.
  step2  reduced instance (16 unknown bits) : exact marginals by enumeration vs
         SCALib BP joint read-out vs product-of-bit-marginals read-out.
  step3  paired ablation, 200 keys, sigma 0.10 / 0.50, A and B:
         HW-only inference vs complete graph (joint read-out) at 10/30/100 iterations.
  step4  one tie convention everywhere; HW-only reference and grid top-k re-checked.

sigma = 0.01 is called NEAR-NOISELESS (not noise-free).

TIE CONVENTION (used for every number produced here):
  ties are broken uniformly at random and reported in expectation.
  For a true value with `better` strictly more likely values and `ties` values of
  exactly equal score (the true one included):
      rank          = better + (ties + 1) / 2          (1-indexed)
      P(rank <= k)  = clip((k - better) / ties, 0, 1)
  "full key" = all 16 bytes have better = 0 and ties = 1 (strict rank 1).
"""
import json
import sys
import time
from math import comb
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scalib.attacks import BPState, GenFactor

import sarry_scalib_core as core
from sarry_scalib_run import seed_for

HERE = Path(__file__).parent
VALS = np.arange(256)
BITS = (VALS[:, None] >> np.arange(8)) & 1
HWV = BITS.sum(axis=1).astype(float)
CONVENTION = "ties broken uniformly at random, reported in expectation: rank = better + (ties+1)/2; P(rank<=k) = clip((k-better)/ties, 0, 1)"


# ------------------------------------------------------------------ rank helpers
def better_ties(score, true_v):
    return int(np.sum(score > score[true_v])), int(np.sum(score == score[true_v]))


def exp_rank(b, t):
    return b + (t + 1) / 2.0


def p_top(b, t, k):
    return float(min(max((k - b) / t, 0.0), 1.0))


def metrics(bt_list):
    """bt_list: list of (better, ties) for key bytes."""
    bt = np.array(bt_list, dtype=float).reshape(-1, 2)
    r = bt[:, 0] + (bt[:, 1] + 1) / 2
    out = {"mean_rank": float(r.mean()), "std_rank": float(r.std())}
    for k in (1, 5, 10):
        out[f"top{k}_pct"] = float(100 * np.mean(np.clip((k - bt[:, 0]) / bt[:, 1], 0, 1)))
    out["bytes_with_ties"] = int(np.sum(bt[:, 1] > 1))
    return out


def hw_only_reference():
    """Exact HW of the key byte known, nothing else. Binomial counts, same tie convention."""
    c = [comb(8, h) for h in range(9)]
    mean = sum(ch / 256 * (ch + 1) / 2 for ch in c)
    var = sum(ch / 256 * sum((r - mean) ** 2 for r in range(1, ch + 1)) / ch for ch in c)
    return {"mean_rank": mean, "std_rank": var ** 0.5,
            **{f"top{k}_pct": 100 * sum(ch / 256 * min(k, ch) / ch for ch in c) for k in (1, 5, 10)}}


# ------------------------------------------------------------------ SCALib BP
def run_bp(seed, sigma, version, n_iters, fixed=None, gens=None):
    """BP on the unmodified graph. fixed: {var_name: bit} -> delta evidence (reduced instance)."""
    key_bits, nonce_bits, chi_in, chi_out, obs = core.simulate(seed, sigma)
    fg = core.get_graph(version)
    pub = {"one": 1}
    if version == "A":
        pub.update({f"n{i}": int(nonce_bits[i]) for i in range(128)})
    if gens is None:
        gens = {f"hk{b}": core.hw_factor(obs["k"][b], sigma) for b in range(16)}
        for g in range(48):
            gens[f"hi{g}"] = core.hw_factor(obs["ci"][g], sigma)
            gens[f"ho{g}"] = core.hw_factor(obs["co"][g], sigma)
        if version == "B":
            gens.update({f"hn{b}": core.hw_factor(obs["n"][b], sigma) for b in range(16)})
    bp = BPState(fg, 1, pub, gens)
    for name, bit in (fixed or {}).items():
        d = np.zeros(2); d[bit] = 1.0
        bp.set_evidence(name, d)
    bp.bp_loopy(n_iters, initialize_states=True)
    return bp, key_bits, nonce_bits, chi_in, chi_out, obs


def readouts(bp, b, obs_kb, sigma):
    """(joint log-score, marginal-product log-score, nan flag) for key byte b."""
    marg = np.array([bp.get_distribution(f"k{8 * b + j}") for j in range(8)], dtype=float)
    msgs = []
    for j in range(8):
        m = bp.get_belief_from_var(f"k{8 * b + j}", f"HK{b}")
        msgs.append(np.full(2, 0.5) if m is None else np.asarray(m, dtype=float).reshape(-1))
    msgs = np.array(msgs)
    nan = bool(not (np.all(np.isfinite(marg)) and np.all(np.isfinite(msgs))))
    with np.errstate(divide="ignore", invalid="ignore"):
        lp_m = np.log(marg[np.arange(8)[None, :], BITS]).sum(axis=1)
        lp_j = -0.5 * ((HWV - obs_kb) / sigma) ** 2 + np.log(msgs[np.arange(8)[None, :], BITS]).sum(axis=1)
    return lp_j, lp_m, nan


def normalise(lp):
    lp = lp - np.max(lp)
    p = np.exp(lp)
    return p / p.sum()


# ------------------------------------------------------------------ step 1 (SAT)
def xor_clauses(lits, rhs):
    """CNF of XOR(lits) = rhs."""
    m = len(lits)
    out = []
    for a in range(1 << m):
        bits = [(a >> i) & 1 for i in range(m)]
        if sum(bits) % 2 != rhs:                       # forbid this assignment
            out.append([-l if bit else l for l, bit in zip(lits, bits)])
    return out


def build_cnf(version, nonce_bits, hw_k, hw_ci, hw_co, hw_n):
    from pysat.card import CardEnc, EncType
    from pysat.formula import IDPool
    pool = IDPool()
    K = [pool.id(f"k{i}") for i in range(128)]
    N = [pool.id(f"n{i}") for i in range(128)] if version == "B" else None
    CI = [pool.id(f"ci{i}") for i in range(384)]
    CO = [pool.id(f"co{i}") for i in range(384)]
    cl = []
    for i in range(384):
        lits = [CI[i]] + [K[j] for j in np.flatnonzero(core.L_K[i])]
        rhs = int(core.C0[i])
        for j in np.flatnonzero(core.L_N[i]):
            if version == "B":
                lits.append(N[j])
            else:
                rhs ^= int(nonce_bits[j])
        cl += xor_clauses(lits, rhs)
    for (i0, i1, i2) in core.COLUMNS:
        for (o, a, b_, c) in ((i0, i0, i1, i2), (i1, i1, i2, i0), (i2, i2, i0, i1)):   # co_o = ci_a ^ (!ci_b & ci_c)
            for va in (0, 1):
                for vb in (0, 1):
                    for vc in (0, 1):
                        out = va ^ ((1 - vb) & vc)
                        cl.append([-CI[a] if va else CI[a], -CI[b_] if vb else CI[b_], -CI[c] if vc else CI[c],
                                   CO[o] if out else -CO[o]])

    def card(lits, h):
        if h == 0:
            return [[-l] for l in lits]
        if h == len(lits):
            return [[l] for l in lits]
        return CardEnc.equals(lits=lits, bound=int(h), vpool=pool, encoding=EncType.seqcounter).clauses

    for b, grp in enumerate(core.KEY_BYTES):
        cl += card([K[j] for j in grp], hw_k[b])
        if version == "B":
            cl += card([N[j] for j in grp], hw_n[b])
    for g, grp in enumerate(core.CHI_BYTES):
        cl += card([CI[j] for j in grp], hw_ci[g])
        cl += card([CO[j] for j in grp], hw_co[g])
    return cl, K, N, pool


def step1_job(args):
    from pysat.solvers import Solver
    version, seed, key_cap = args
    t0 = time.time()
    key_bits, nonce_bits, chi_in, chi_out, _ = core.simulate(seed, 1.0)
    hw_k = [int(key_bits[g].sum()) for g in core.KEY_BYTES]
    hw_n = [int(nonce_bits[g].sum()) for g in core.KEY_BYTES]
    hw_ci = [int(chi_in[g].sum()) for g in core.CHI_BYTES]
    hw_co = [int(chi_out[g].sum()) for g in core.CHI_BYTES]

    # (a) SCALib's own check with exact indicator factors on the true assignment
    ind = lambda h: GenFactor.dense((core.HW8 == h).astype(np.float64))
    gens = {f"hk{b}": ind(hw_k[b]) for b in range(16)}
    for g in range(48):
        gens[f"hi{g}"] = ind(hw_ci[g]); gens[f"ho{g}"] = ind(hw_co[g])
    if version == "B":
        gens.update({f"hn{b}": ind(hw_n[b]) for b in range(16)})
    var = {f"k{i}": int(key_bits[i]) for i in range(128)}
    pub = {"one": 1}
    (pub if version == "A" else var).update({f"n{i}": int(nonce_bits[i]) for i in range(128)})
    for i in range(384):
        var.update({f"ci{i}": int(chi_in[i]), f"na{i}": 1 - int(chi_in[i]),
                    f"t{i}": int(chi_in[i] ^ chi_out[i]), f"co{i}": int(chi_out[i])})
    try:
        core.get_graph(version).sanity_check(pub, var, gens)
        scalib_ok = True
    except Exception as e:                                           # noqa
        scalib_ok = str(e)[:200]
    indicator_like = 1.0                                             # product of all indicator factors at the truth
    for b in range(16):
        indicator_like *= float(key_bits[core.KEY_BYTES[b]].sum() == hw_k[b])

    cl, K, N, pool = build_cnf(version, nonce_bits, hw_k, hw_ci, hw_co, hw_n)
    s = Solver(name="cadical153", bootstrap_with=cl)
    true_assump = [K[i] if key_bits[i] else -K[i] for i in range(128)]
    if version == "B":
        true_assump += [N[i] if nonce_bits[i] else -N[i] for i in range(128)]
    true_sat = bool(s.solve(assumptions=true_assump))

    def key_of(model):
        ms = set(l for l in model if l > 0)
        return tuple(1 if K[i] in ms else 0 for i in range(128))

    # Budgeted SAT calls: True = consistent assignment found, False = proven impossible,
    # None = undecided within the budget (conflict limit, or the per-trial time limit).
    T_LIMIT, CONF = 240.0, 300000

    def lim(solver, assumptions=()):
        if time.time() - t0 > T_LIMIT:
            return None
        solver.conf_budget(CONF)
        return solver.solve_limited(assumptions=list(assumptions))

    # whole keys consistent with the exact observations (enumerated up to key_cap)
    keys = set()
    capped = False
    whole_key_status = "exact"
    while True:
        r = lim(s)
        if r is None:
            capped = True; whole_key_status = "undecided (budget) - lower bound"
            break
        if not r:
            break
        k = key_of(s.get_model())
        keys.add(k)
        s.add_clause([-K[i] if k[i] else K[i] for i in range(128)])
        if len(keys) >= key_cap:
            capped = True; whole_key_status = "cap reached - lower bound"
            break
    per_byte, per_byte_exact = [], []
    if not capped:
        for b in range(16):
            per_byte.append(len({tuple(k[8 * b:8 * b + 8]) for k in keys}))
            per_byte_exact.append(True)
    else:                                           # per-byte counting with a fresh solver and selectors
        s.delete()
        s = Solver(name="cadical153", bootstrap_with=cl)
        found = [{tuple(k[8 * b:8 * b + 8]) for k in keys} for b in range(16)]
        for b in range(16):
            sel = pool.id(f"sel{b}")
            for v in found[b]:
                s.add_clause([-sel] + [-K[8 * b + j] if v[j] else K[8 * b + j] for j in range(8)])
            exact = False
            while len(found[b]) < comb(8, hw_k[b]):
                r = lim(s, [sel])
                if r is None:
                    break
                if not r:
                    exact = True
                    break
                k = key_of(s.get_model())
                for bb in range(16):
                    found[bb].add(tuple(k[8 * bb:8 * bb + 8]))
                v = tuple(k[8 * b:8 * b + 8])
                s.add_clause([-sel] + [-K[8 * b + j] if v[j] else K[8 * b + j] for j in range(8)])
            if len(found[b]) == comb(8, hw_k[b]):
                exact = True                        # every value allowed by the byte's own HW is consistent
            per_byte_exact.append(exact)
        per_byte = [len(found[b]) for b in range(16)]   # values PROVEN consistent (a model was found)
    s.delete()
    return {"version": version, "seed": seed, "scalib_sanity_check_true_assignment": scalib_ok,
            "indicator_likelihood_of_truth": indicator_like, "sat_true_assignment_satisfiable": true_sat,
            "consistent_keys": len(keys), "consistent_keys_capped": capped,
            "true_key_in_enumerated_set": tuple(int(x) for x in key_bits) in keys,
            "key_byte_hw": hw_k, "own_hw_count_per_byte": [comb(8, h) for h in hw_k],
            "whole_key_count_status": whole_key_status,
            "consistent_values_per_byte": per_byte, "per_byte_count_is_exact": per_byte_exact, "runtime_s": time.time() - t0}


def step1(n=20, key_cap=2000):
    print("=== STEP 1: exact-HW (indicator) test ===", flush=True)
    # whole-key enumeration cap: 2000 for A; 50 for B (B has many consistent keys, so it switches
    # to exact per-byte counting, which does not need the whole-key list)
    out = {"n_trials": n, "key_cap": {"A": key_cap, "B": 50}, "trials": []}
    path = HERE / "sarry_scalib_diag2_step1_trials.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    have = {(r["version"], r["seed"]) for r in rows}
    todo = [(v, 44000 + s, key_cap if v == "A" else 50) for v in ("A", "B") for s in range(n) if (v, 44000 + s) not in have]
    with open(path, "a") as f, Pool(4) as pool:
        for r in pool.imap_unordered(step1_job, todo, chunksize=1):
            f.write(json.dumps(r) + "\n"); f.flush()
            rows.append(r)
            print(f"  done {r['version']} seed {r['seed']} in {r['runtime_s']:.0f}s", flush=True)
    rows = [r for r in rows if r["seed"] < 44000 + n]
    rows.sort(key=lambda r: (r["version"], r["seed"]))
    out["trials"] = rows
    for v in ("A", "B"):
        sel = [r for r in rows if r["version"] == v]
        own = np.array([r["own_hw_count_per_byte"] for r in sel])
        cons = np.array([r["consistent_values_per_byte"] for r in sel])
        print(f"version {v}: true assignment accepted by SCALib sanity_check (indicator factors): "
              f"{sum(r['scalib_sanity_check_true_assignment'] is True for r in sel)}/{n}; "
              f"satisfiable in the SAT model: {sum(r['sat_true_assignment_satisfiable'] for r in sel)}/{n}")
        print(f"   whole keys consistent with the exact observations: "
              f"{[str(r['consistent_keys']) + ('+' if r['consistent_keys_capped'] else '') for r in sel]}")
        print(f"   trials where the key is UNIQUELY identified: {sum((not r['consistent_keys_capped']) and r['consistent_keys'] == 1 for r in sel)}/{n}")
        print(f"   key-byte values consistent, own HW only : mean {own.mean():.2f} (min {own.min()}, max {own.max()})")
        ex = np.array([r["per_byte_count_is_exact"] for r in sel])
        print(f"   key-byte values PROVEN consistent with all exact observations: mean {cons.mean():.2f} "
              f"(min {cons.min()}, max {cons.max()}); count is exact for {int(ex.sum())}/{ex.size} bytes, a lower bound for the rest")
        print(f"   bytes uniquely identified (exact count = 1): {int(((cons == 1) & ex).sum())}/{cons.size}; "
              f"bytes where EVERY value allowed by the own HW stays consistent: {int((cons == own).sum())}/{cons.size}")
        print(f"   ratio consistent / own-HW count, mean over bytes: {np.mean(cons / own):.3f}")
        print(f"   runtime per trial: mean {np.mean([r['runtime_s'] for r in sel]):.1f}s, max {max(r['runtime_s'] for r in sel):.1f}s", flush=True)
    (HERE / "sarry_scalib_diag2_step1.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------ step 2
LKf = core.L_K.astype(np.float32)
LNf = core.L_N.astype(np.float32)
COLS = np.array(core.COLUMNS)
CHIB = np.array(core.CHI_BYTES)
KEYB = np.array(core.KEY_BYTES)


def exact_enumeration(key_bits, nonce_bits, obs, sigma, version, unk_k, unk_n):
    """Exact posterior over the unknown bits (all others fixed to the truth)."""
    u = len(unk_k) + len(unk_n)
    A = ((np.arange(1 << u)[:, None] >> np.arange(u)) & 1).astype(np.float32)
    K = np.tile(key_bits.astype(np.float32), (1 << u, 1)); K[:, unk_k] = A[:, :len(unk_k)]
    Nn = np.tile(nonce_bits.astype(np.float32), (1 << u, 1))
    if unk_n:
        Nn[:, unk_n] = A[:, len(unk_k):]
    ci = (np.rint(K @ LKf.T + Nn @ LNf.T).astype(np.int64) + core.C0[None, :]) % 2
    co = np.zeros_like(ci)
    a0, a1, a2 = ci[:, COLS[:, 0]], ci[:, COLS[:, 1]], ci[:, COLS[:, 2]]
    co[:, COLS[:, 0]] = a0 ^ ((1 - a1) & a2)
    co[:, COLS[:, 1]] = a1 ^ ((1 - a2) & a0)
    co[:, COLS[:, 2]] = a2 ^ ((1 - a0) & a1)
    ll = -0.5 * (((ci[:, CHIB].sum(axis=2) - np.array(obs["ci"])) / sigma) ** 2).sum(axis=1)
    ll += -0.5 * (((co[:, CHIB].sum(axis=2) - np.array(obs["co"])) / sigma) ** 2).sum(axis=1)
    ll += -0.5 * (((K[:, KEYB].sum(axis=2) - np.array(obs["k"])) / sigma) ** 2).sum(axis=1)
    if version == "B":
        ll += -0.5 * (((Nn[:, KEYB].sum(axis=2) - np.array(obs["n"])) / sigma) ** 2).sum(axis=1)
    post = np.exp(ll - ll.max()); post /= post.sum()
    return post, K


def step2_job(args):
    version, sigma, seed = args
    key_bits, nonce_bits, chi_in, chi_out, obs = core.simulate(seed, sigma)
    rng = np.random.default_rng(seed + 7)
    if version == "A":
        kbytes = sorted(rng.choice(16, 2, replace=False).tolist()); nbytes = []
    else:
        kbytes = [int(rng.integers(16))]; nbytes = [int(rng.integers(16))]
    unk_k = [8 * b + j for b in kbytes for j in range(8)]
    unk_n = [8 * b + j for b in nbytes for j in range(8)]
    post, K = exact_enumeration(key_bits, nonce_bits, obs, sigma, version, unk_k, unk_n)
    fixed = {f"k{i}": int(key_bits[i]) for i in range(128) if i not in unk_k}
    if version == "B":
        fixed.update({f"n{i}": int(nonce_bits[i]) for i in range(128) if i not in unk_n})
    res = {"version": version, "sigma": sigma, "seed": seed, "unknown_key_bytes": kbytes,
           "unknown_nonce_bytes": nbytes, "bytes": []}
    exact = {}
    for b in kbytes:
        vals = (K[:, 8 * b:8 * b + 8] @ (1 << np.arange(8)).astype(np.float32)).astype(int)
        exact[b] = np.bincount(vals, weights=post, minlength=256)
    for n_iters in (10, 30, 100):
        bp = run_bp(seed, sigma, version, n_iters, fixed=fixed)[0]
        for b in kbytes:
            true_v = int((key_bits[8 * b:8 * b + 8] * (1 << np.arange(8))).sum())
            lp_j, lp_m, nan = readouts(bp, b, obs["k"][b], sigma)
            e = exact[b]
            row = {"byte": b, "n_iters": n_iters, "nan": nan, "exact_p_true": float(e[true_v]),
                   "exact_bt": better_ties(e, true_v)}
            if not nan:
                pj, pm = normalise(lp_j), normalise(lp_m)
                row.update(tv_joint=float(0.5 * np.abs(pj - e).sum()), tv_marginal=float(0.5 * np.abs(pm - e).sum()),
                           joint_bt=better_ties(lp_j, true_v), marginal_bt=better_ties(lp_m, true_v),
                           joint_p_true=float(pj[true_v]), marginal_p_true=float(pm[true_v]))
            res["bytes"].append(row)
    return res


def step2(n=20):
    print("=== STEP 2: reduced instance (16 unknown bits), exact enumeration vs SCALib BP read-outs ===")
    print("    A: 2 unknown key bytes, other 112 key bits fixed to the truth (delta evidence in SCALib)")
    print("    B: 1 unknown key byte + 1 unknown nonce byte, all other key and nonce bits fixed to the truth", flush=True)
    jobs = [(v, s, 43000 + t) for v in ("A", "B") for s in (0.01, 0.10, 0.50, 1.00, 2.00) for t in range(n)]
    with Pool(3) as pool:
        rows = pool.map(step2_job, jobs, chunksize=1)
    out = {"trials": rows, "summary": {}, "tie_convention": CONVENTION}
    for v in ("A", "B"):
        for s in (0.01, 0.10, 0.50, 1.00, 2.00):
            for n_iters in (10, 30, 100):
                bs = [b for r in rows if r["version"] == v and r["sigma"] == s for b in r["bytes"] if b["n_iters"] == n_iters]
                ok = [b for b in bs if not b["nan"]]
                summ = {"bytes": len(bs), "nan_bytes": len(bs) - len(ok),
                        "exact_mean_rank": metrics([b["exact_bt"] for b in bs])["mean_rank"],
                        "exact_top1_pct": metrics([b["exact_bt"] for b in bs])["top1_pct"],
                        "exact_mean_p_true": float(np.mean([b["exact_p_true"] for b in bs]))}
                line = (f"ver={v} sigma={s:4.2f} iters={n_iters:3d}: EXACT mean rank {summ['exact_mean_rank']:6.2f} "
                        f"top1 {summ['exact_top1_pct']:5.1f}% P(true) {summ['exact_mean_p_true']:.3f}")
                if ok:
                    summ.update(joint_mean_rank=metrics([b["joint_bt"] for b in ok])["mean_rank"],
                                joint_top1_pct=metrics([b["joint_bt"] for b in ok])["top1_pct"],
                                marginal_mean_rank=metrics([b["marginal_bt"] for b in ok])["mean_rank"],
                                marginal_top1_pct=metrics([b["marginal_bt"] for b in ok])["top1_pct"],
                                tv_joint_mean=float(np.mean([b["tv_joint"] for b in ok])),
                                tv_joint_max=float(np.max([b["tv_joint"] for b in ok])),
                                tv_marginal_mean=float(np.mean([b["tv_marginal"] for b in ok])),
                                tv_marginal_max=float(np.max([b["tv_marginal"] for b in ok])))
                    line += (f" | JOINT rank {summ['joint_mean_rank']:6.2f} top1 {summ['joint_top1_pct']:5.1f}% "
                             f"TV mean {summ['tv_joint_mean']:.3f} max {summ['tv_joint_max']:.3f}"
                             f" | MARGINAL rank {summ['marginal_mean_rank']:6.2f} top1 {summ['marginal_top1_pct']:5.1f}% "
                             f"TV mean {summ['tv_marginal_mean']:.3f} max {summ['tv_marginal_max']:.3f}")
                line += f" | NaN bytes {summ['nan_bytes']}/{summ['bytes']}"
                out["summary"][f"{v}_{s}_{n_iters}"] = summ
                print(line, flush=True)
    (HERE / "sarry_scalib_diag2_step2.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------ step 3
def step3_job(args):
    version, sigma, k, n_iters = args
    seed = seed_for(sigma, k)
    t0 = time.perf_counter()
    bp, key_bits, nonce_bits, chi_in, chi_out, obs = run_bp(seed, sigma, version, n_iters)
    joint, hw_only, nan = [], [], False
    for b in range(16):
        true_v = int((key_bits[8 * b:8 * b + 8] * (1 << np.arange(8))).sum())
        lp_j, lp_m, nb = readouts(bp, b, obs["k"][b], sigma)
        nan = nan or nb
        joint.append(better_ties(lp_j, true_v))
        hw_only.append(better_ties(-0.5 * ((HWV - obs["k"][b]) / sigma) ** 2, true_v))
    return {"version": version, "sigma": sigma, "k": k, "seed": seed, "n_iters": n_iters, "nan": nan,
            "joint_bt": None if nan else joint, "hw_only_bt": hw_only, "runtime_s": time.perf_counter() - t0}


def per_key(bt):
    bt = np.array(bt, dtype=float)
    r = bt[:, 0] + (bt[:, 1] + 1) / 2
    return [r.mean()] + [100 * np.mean(np.clip((k - bt[:, 0]) / bt[:, 1], 0, 1)) for k in (1, 5, 10)]


def boot_ci(d, rng, n=10000):
    d = np.asarray(d)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)          # (n resamples, 4 metrics): mean over the resampled keys
    return [float(d.mean(axis=0)[i]) for i in range(d.shape[1])], \
           [[float(np.percentile(m[:, i], 2.5)), float(np.percentile(m[:, i], 97.5))] for i in range(d.shape[1])]


def step3(n_keys=200):
    print("=== STEP 3: paired ablation on the same noisy observations (200 keys = first 200 grid seeds) ===", flush=True)
    path = HERE / "sarry_scalib_diag2_step3_trials.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done[(r["version"], r["sigma"], r["k"], r["n_iters"])] = r
    todo = [(v, s, k, it) for it in (10, 30, 100) for v in ("A", "B") for s in (0.10, 0.50) for k in range(n_keys)
            if (v, s, k, it) not in done]
    with open(path, "a") as f, Pool(4) as pool:
        for r in pool.imap_unordered(step3_job, todo, chunksize=4):
            f.write(json.dumps(r) + "\n"); f.flush()
            done[(r["version"], r["sigma"], r["k"], r["n_iters"])] = r
    rng = np.random.default_rng(12345)
    names = ["mean_rank", "top1_pct", "top5_pct", "top10_pct"]
    out = {"n_keys": n_keys, "tie_convention": CONVENTION, "bootstrap": "10000 resamples over keys, percentile 95% CI, rng seed 12345",
           "settings": {}}
    for v in ("A", "B"):
        for s in (0.10, 0.50):
            rows10 = [done[(v, s, k, 10)] for k in range(n_keys)]
            hw = metrics([bt for r in rows10 for bt in r["hw_only_bt"]])
            entry = {"hw_only": hw}
            print(f"ver={v} sigma={s:4.2f} HW-only (no graph)     : mean {hw['mean_rank']:6.2f} top1 {hw['top1_pct']:5.2f}% "
                  f"top5 {hw['top5_pct']:5.2f}% top10 {hw['top10_pct']:5.2f}%  NaN 0/{n_keys}")
            for it in (10, 30, 100):
                rows = [done[(v, s, k, it)] for k in range(n_keys)]
                ok = [r for r in rows if not r["nan"]]
                e = {"nan_trials": n_keys - len(ok), "valid_trials": len(ok),
                     "mean_runtime_s_per_key": float(np.mean([r["runtime_s"] for r in rows]))}
                line = f"ver={v} sigma={s:4.2f} graph joint, {it:3d} iters : "
                if ok:
                    m = metrics([bt for r in ok for bt in r["joint_bt"]])
                    e.update(m)
                    e["full_key"] = int(sum(all(bt[0] == 0 and bt[1] == 1 for bt in r["joint_bt"]) for r in ok))
                    d = [np.array(per_key(r["joint_bt"])) - np.array(per_key(r["hw_only_bt"])) for r in ok]
                    mean_d, ci = boot_ci(d, rng)
                    e["paired_diff_graph_minus_hw_only"] = {n_: {"mean": mean_d[i], "ci95": ci[i]} for i, n_ in enumerate(names)}
                    e["hw_only_on_same_valid_keys"] = metrics([bt for r in ok for bt in r["hw_only_bt"]])
                    line += (f"mean {m['mean_rank']:6.2f} top1 {m['top1_pct']:5.2f}% top5 {m['top5_pct']:5.2f}% "
                             f"top10 {m['top10_pct']:5.2f}%  full key {e['full_key']}/{len(ok)}  NaN {n_keys - len(ok)}/{n_keys}"
                             f"  ties {m['bytes_with_ties']}\n      paired diff vs HW-only (valid keys): "
                             + "; ".join(f"{n_} {mean_d[i]:+.2f} [{ci[i][0]:+.2f}, {ci[i][1]:+.2f}]" for i, n_ in enumerate(names)))
                else:
                    line += f"no valid trial, NaN {n_keys}/{n_keys}"
                entry[f"graph_joint_{it}"] = e
                print(line, flush=True)
            for it in (30, 100):                                   # iterations vs 10 iterations, keys valid in both
                pairs = [(done[(v, s, k, it)], done[(v, s, k, 10)]) for k in range(n_keys)]
                pairs = [(a, b) for a, b in pairs if not a["nan"] and not b["nan"]]
                if pairs:
                    d = [np.array(per_key(a["joint_bt"])) - np.array(per_key(b["joint_bt"])) for a, b in pairs]
                    mean_d, ci = boot_ci(d, rng)
                    entry[f"paired_diff_{it}_minus_10_iters"] = {"n_pairs": len(pairs),
                        **{n_: {"mean": mean_d[i], "ci95": ci[i]} for i, n_ in enumerate(names)}}
                    print(f"      paired diff {it} - 10 iterations ({len(pairs)} keys): "
                          + "; ".join(f"{n_} {mean_d[i]:+.2f} [{ci[i][0]:+.2f}, {ci[i][1]:+.2f}]" for i, n_ in enumerate(names)), flush=True)
            out["settings"][f"{v}_{s}"] = entry
    (HERE / "sarry_scalib_diag2_step3.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------ step 4
def step4():
    print("=== STEP 4: one tie convention ===")
    print("convention:", CONVENTION)
    ref = hw_only_reference()
    print("HW-only reference (exact HW, binomial counts): " + " ".join(f"{k}={v:.2f}" for k, v in ref.items()))
    out = {"tie_convention": CONVENTION, "hw_only_reference": ref,
           "random_guess_reference": {"mean_rank": 128.5, "std_rank": ((256 ** 2 - 1) / 12) ** 0.5,
                                      "top1_pct": 100 / 256, "top5_pct": 500 / 256, "top10_pct": 1000 / 256}}
    # grid: stored value is better + (ties-1)/2 (0-indexed mid-rank). Odd (ties-1) shows up as a .5 value.
    grid = [json.loads(l) for l in (HERE / "sarry_scalib_joint_grid_trials.jsonl").read_text().splitlines()]
    frac = sum(1 for r in grid if not r["nan"] for x in r["ranks0_joint"] if x != int(x))
    total = sum(16 for r in grid if not r["nan"])
    out["grid_joint_ranks_with_half_integer_value"] = [frac, total]
    print(f"grid (16000 trials): joint-read-out ranks with a half-integer value (= tie of even size): {frac}/{total}")
    # step-3 runs at 10 iterations are the first 200 grid keys at sigma 0.10 / 0.50: compare and count ties exactly
    p3 = HERE / "sarry_scalib_diag2_step3_trials.jsonl"
    if p3.exists():
        g = {(r["version"], r["sigma"], r["seed"]): r for r in grid}
        n_cmp = n_same = n_ties = n_bytes = 0
        for line in p3.read_text().splitlines():
            r = json.loads(line)
            if r["n_iters"] != 10 or r["nan"]:
                continue
            gr = g[(r["version"], r["sigma"], r["seed"])]
            mid = [bt[0] + (bt[1] - 1) / 2 for bt in r["joint_bt"]]
            n_cmp += 1
            n_same += int(mid == gr["ranks0_joint"])
            n_ties += sum(bt[1] > 1 for bt in r["joint_bt"]); n_bytes += 16
        out["step3_vs_grid_identical_trials"] = [n_same, n_cmp]
        out["step3_10iter_joint_bytes_with_ties"] = [n_ties, n_bytes]
        print(f"step-3 10-iteration runs vs the grid's stored ranks (same seeds): {n_same}/{n_cmp} trials identical")
        print(f"ties in the joint read-out, counted exactly on those runs: {n_ties}/{n_bytes} bytes")
    (HERE / "sarry_scalib_diag2_step4.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "step1":
        step1(int(sys.argv[2]) if len(sys.argv) > 2 else 20, int(sys.argv[3]) if len(sys.argv) > 3 else 2000)
    elif which == "step2":
        step2(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif which == "step3":
        step3()
    elif which == "step4":
        step4()
