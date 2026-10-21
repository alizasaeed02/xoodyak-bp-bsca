"""
Final reporting checks for the byte-level rank experiment (SCALib 0.6.4, graph =
sarry_scalib_core.py unmodified). No manuscript edits.

  sat     item 1: SAT status, blocking clause and witness keys per instance (20 x A/B)
  ties    item 3: rerun of the grid rows with (better, ties) stored per key byte
          (the first grid stored only the expected rank, for every row)
  report  items 2-4: NaN accounting for the ablation, corrected table, list of changes

Tie convention: ties broken uniformly at random, reported in expectation:
  rank = better + (ties+1)/2 ;  P(rank <= k) = clip((k - better)/ties, 0, 1) per byte.
"Std" = standard deviation of the per-byte EXPECTED ranks (not of realised random ranks).
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import sarry_scalib_core as core
import sarry_scalib_diag2 as d2
from linear_theta_CORRECTED_init import key_bits_to_state, state_to_chi_input_bits
from sarry_scalib_run import seed_for

HERE = Path(__file__).parent
SIGMAS = [0.01, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00]
N_KEYS = 1000
TIES_JSONL = HERE / "sarry_scalib_joint_grid_ties_trials.jsonl"


# ----------------------------------------------------------------------------- item 1
def hws_of(key_bits, nonce_bits):
    """All byte Hamming weights from a key and a nonce, via xoodoo_ref (not via the L matrices)."""
    nonce_bytes = bytes(np.packbits(np.array(nonce_bits, dtype=np.uint8), bitorder="little"))
    ci = state_to_chi_input_bits(key_bits_to_state(np.array(key_bits, dtype=np.uint8), nonce_bytes))
    co = np.zeros(384, dtype=np.uint8)
    for (i0, i1, i2) in core.COLUMNS:
        co[i0], co[i1], co[i2] = core.chi3(int(ci[i0]), int(ci[i1]), int(ci[i2]))
    kb, nb = np.array(key_bits), np.array(nonce_bits)
    return ([int(kb[g].sum()) for g in core.KEY_BYTES], [int(ci[g].sum()) for g in core.CHI_BYTES],
            [int(co[g].sum()) for g in core.CHI_BYTES], [int(nb[g].sum()) for g in core.KEY_BYTES])


def hexs(bits):
    return bytes(np.packbits(np.array(bits, dtype=np.uint8), bitorder="little")).hex()


def sat_job(args):
    from pysat.solvers import Solver
    version, seed = args
    t0 = time.time()
    key_bits, nonce_bits, chi_in, chi_out, _ = core.simulate(seed, 1.0)
    hk, hci, hco, hn = hws_of(key_bits, nonce_bits)
    cl, K, N, pool = d2.build_cnf(version, nonce_bits, hk, hci, hco, hn)
    s = Solver(name="cadical153", bootstrap_with=cl)

    def extract(model):
        ms = set(l for l in model if l > 0)
        k = [1 if K[i] in ms else 0 for i in range(128)]
        n = [1 if N[i] in ms else 0 for i in range(128)] if version == "B" else [int(x) for x in nonce_bits]
        return k, n

    def verify(k, n):
        a = hws_of(k, n)
        ok = a[0] == hk and a[1] == hci and a[2] == hco
        return bool(ok and (version == "A" or a[3] == hn))

    res = {"version": version, "seed": seed, "solver": "CaDiCaL 1.5.3 (python-sat)",
           "true_key_hex": hexs(key_bits),
           "blocking_clause": "one clause over the 128 KEY variables only (at least one key bit differs from witness 1); "
                              "nonce variables and all auxiliary variables are left free"}
    r1 = s.solve()
    res["search1_status"] = "SAT" if r1 else "UNSAT"
    k1, n1 = extract(s.get_model())
    res.update(witness1_key_hex=hexs(k1), witness1_is_true_key=bool(k1 == [int(x) for x in key_bits]),
               witness1_verified_by_forward_computation=verify(k1, n1))
    if version == "B":
        res["witness1_nonce_hex"] = hexs(n1)
    block = [-K[i] if k1[i] else K[i] for i in range(128)]
    assert len(block) == 128 and set(abs(l) for l in block) == set(K)
    s.add_clause(block)
    s.conf_budget(300_000)
    r2 = s.solve_limited()
    method = "direct search with witness-1 key blocked (budget 300,000 conflicts)"
    k2 = None
    if r2:
        k2, n2 = extract(s.get_model())
    elif r2 is None:                 # undecided: try forcing each key bit away from witness 1 (still key-only)
        for i in range(128):
            if time.time() - t0 > 180:           # wall-clock cap per instance for the fallback
                break
            s.conf_budget(20_000)
            r = s.solve_limited(assumptions=[-K[i] if k1[i] else K[i]])
            if r:
                k2, n2 = extract(s.get_model()); r2 = True
                method = f"search with witness-1 key blocked and key bit {i} forced to differ (budget 20,000 conflicts per bit, 180 s cap per instance)"
                break
    res["search2_status"] = "SAT" if r2 else ("UNSAT" if r2 is False else "UNDECIDED (budget exhausted)")
    res["search2_method"] = method
    if k2 is not None:
        res.update(witness2_key_hex=hexs(k2), witness2_differs_from_witness1_in_key_bits=int(np.sum(np.array(k1) != np.array(k2))),
                   witness2_verified_by_forward_computation=verify(k2, n2))
        if version == "B":
            res["witness2_nonce_hex"] = hexs(n2)
    res["runtime_s"] = time.time() - t0
    s.delete()
    return res


def sat():
    jobs = [(v, 44000 + s) for v in ("A", "B") for s in range(20)]
    rows = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(sat_job, jobs, chunksize=1):
            rows.append(r)
            print(f"  finished {r['version']} {r['seed']}: search2 {r['search2_status']} [{r['runtime_s']:.0f}s]", flush=True)
    rows.sort(key=lambda r: (r["version"], r["seed"]))
    (HERE / "sarry_scalib_final_checks_sat.json").write_text(json.dumps(rows, indent=1))
    for v in ("A", "B"):
        sel = [r for r in rows if r["version"] == v]
        print(f"--- version {v} ---")
        for r in sel:
            line = (f"seed {r['seed']}: search1 {r['search1_status']} (true key: {r['witness1_is_true_key']}, "
                    f"verified: {r['witness1_verified_by_forward_computation']}); search2 {r['search2_status']}")
            if "witness2_key_hex" in r:
                line += (f" key2 differs in {r['witness2_differs_from_witness1_in_key_bits']} bits, "
                         f"verified: {r['witness2_verified_by_forward_computation']}")
            print(line + f"  [{r['runtime_s']:.0f}s]")
        from collections import Counter
        print("search2 status counts:", dict(Counter(r["search2_status"] for r in sel)))
    print("SAT DONE", flush=True)


# ----------------------------------------------------------------------------- item 3
def ties_run():
    done = set()
    if TIES_JSONL.exists():
        for line in TIES_JSONL.read_text().splitlines():
            r = json.loads(line)
            done.add((r["version"], r["sigma"], r["k"]))
    t_all = time.time()
    with open(TIES_JSONL, "a") as f, Pool(4) as pool:
        for version in ("A", "B"):
            for sigma in SIGMAS:
                todo = [(version, sigma, k, 10) for k in range(N_KEYS) if (version, sigma, k) not in done]
                for r in pool.imap_unordered(d2.step3_job, todo, chunksize=4):
                    f.write(json.dumps(r) + "\n"); f.flush()
                print(f"TIES ver={version} sigma={sigma}: {len(todo)} new trials (total {time.time()-t_all:.0f}s)", flush=True)
    print("TIES FINISHED", flush=True)


def row_stats(rows):
    ok = [r for r in rows if not r["nan"]]
    bt = np.array([b for r in ok for b in r["joint_bt"]], dtype=float)
    er = bt[:, 0] + (bt[:, 1] + 1) / 2
    out = {"keys": len(rows), "nan_trials": len(rows) - len(ok), "valid_trials": len(ok), "key_bytes": int(len(er)),
           "mean_rank": float(er.mean()), "std_of_expected_ranks": float(er.std()),
           "full_key": int(sum(all(b[0] == 0 and b[1] == 1 for b in r["joint_bt"]) for r in ok)),
           "bytes_with_ties": int(np.sum(bt[:, 1] > 1)), "largest_tie": int(bt[:, 1].max())}
    for k in (1, 5, 10):
        out[f"top{k}_pct"] = float(100 * np.mean(np.clip((k - bt[:, 0]) / bt[:, 1], 0, 1)))
        out[f"top{k}_pct_old_threshold_on_expected_rank"] = float(100 * np.mean(er <= k))
    return out


def report():
    # ---- item 3/4: corrected table
    rows = [json.loads(l) for l in TIES_JSONL.read_text().splitlines()]
    old_trials = {(r["version"], r["sigma"], r["seed"]): r for r in
                  (json.loads(l) for l in (HERE / "sarry_scalib_joint_grid_trials.jsonl").read_text().splitlines())}
    old_sum = json.loads((HERE / "sarry_scalib_joint_grid_summary.json").read_text())["summary"]
    same = sum(1 for r in rows if not r["nan"] and
               [b[0] + (b[1] - 1) / 2 for b in r["joint_bt"]] == old_trials[(r["version"], r["sigma"], r["seed"])]["ranks0_joint"])
    print(f"rerun vs first grid: expected ranks identical in {same}/{len(rows)} trials")
    summ, changes = {}, []
    head = ["Version", "sigma", "Mean rank", "Std", "Top-1 %", "Top-5 %", "Top-10 %", "Full key", "NaN trials"]
    body = []
    for version, name in (("A", "A (nonce known)"), ("B", "B (nonce unknown)")):
        for sigma in SIGMAS:
            sel = [r for r in rows if r["version"] == version and r["sigma"] == sigma]
            assert sorted(r["seed"] for r in sel) == [seed_for(sigma, k) for k in range(N_KEYS)]
            s = row_stats(sel)
            summ[f"{version}_{sigma}"] = s
            o = old_sum[f"{version}_{sigma}"]["joint"]
            new_cells = [f"{s['mean_rank']:.2f}", f"{s['std_of_expected_ranks']:.2f}", f"{s['top1_pct']:.2f}",
                         f"{s['top5_pct']:.2f}", f"{s['top10_pct']:.2f}", f"{s['full_key']}/{s['valid_trials']}", f"{s['nan_trials']}"]
            old_cells = [f"{o['mean_rank']:.2f}", f"{o['std_rank']:.2f}", f"{o['top1_pct']:.2f}", f"{o['top5_pct']:.2f}",
                         f"{o['top10_pct']:.2f}", f"{o['full_key']}/{o['valid_trials']}", f"{o['nan_trials']}"]
            for col, a, b in zip(head[2:], old_cells, new_cells):
                if a != b:
                    changes.append(f"{name}, sigma {sigma:.2f}, {col}: {a} -> {b}")
            lab = f"{sigma:.2f}" + (" (near-noiseless)" if sigma == 0.01 else "")
            body.append([name, lab] + new_cells)
            print(f"{version} {sigma:.2f}: bytes with ties {s['bytes_with_ties']}/{s['key_bytes']} (largest tie {s['largest_tie']})")
    ref_hw = d2.hw_only_reference()
    refs = [["Reference: random guessing", "-", "128.50", f"{((256 ** 2 - 1) / 12) ** 0.5:.2f}", f"{100/256:.2f}",
             f"{500/256:.2f}", f"{1000/256:.2f}", "-", "-"],
            ["Reference: key-byte HW only", "-", f"{ref_hw['mean_rank']:.2f}", f"{ref_hw['std_rank']:.2f}",
             f"{ref_hw['top1_pct']:.2f}", f"{ref_hw['top5_pct']:.2f}", f"{ref_hw['top10_pct']:.2f}", "-", "-"]]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in body + refs]
    (HERE / "sarry_scalib_joint_grid_table_v2.md").write_text("\n".join(md) + "\n")
    (HERE / "sarry_scalib_joint_grid_table_v2.csv").write_text("\n".join(",".join(r) for r in [head] + body + refs) + "\n")
    tex = [r"% Generated by sarry_scalib_final_checks.py -- SCALib 0.6.4, 10 BP iterations, joint read-out, 1000 keys per row.",
           r"% Ties broken uniformly at random, in expectation. Std = standard deviation of the per-byte expected ranks.",
           r"% sigma = 0.01 is the near-noiseless row.",
           r"\begin{table}[htbp]",
           r"\caption{Byte-level BP on the first round of Xoodyak (simulated byte Hamming-weight leakage, 1000 keys per row, "
           r"10 loopy-BP iterations in SCALib): rank of the correct key byte. Rank 1 is best; ``Full key'' counts keys with "
           r"all 16 bytes at rank 1.}", r"\label{tab:xoodyak_byte_rank}",
           r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
           r"$\sigma$ & Mean rank & Std & Top-1 (\%) & Top-5 (\%) & Top-10 (\%) & Full key & NaN \\"]
    for name, lo in (("Nonce known", 0), ("Nonce unknown", len(SIGMAS))):
        tex += [r"\midrule", rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{name}}}}} \\"]
        for i, sigma in enumerate(SIGMAS):
            tex.append(" & ".join([f"{sigma:.2f}" + (r"$^{a}$" if sigma == 0.01 else "")] + body[lo + i][2:]) + r" \\")
    tex += [r"\midrule", r"\multicolumn{8}{@{}l}{\textit{References (no attack run)}} \\",
            " & ".join(["Random guess"] + refs[0][2:7] + ["--", "--"]) + r" \\",
            " & ".join(["Key-byte HW only"] + refs[1][2:7] + ["--", "--"]) + r" \\",
            r"\botrule", r"\end{tabular}", r"\footnotetext{$^{a}$ Near-noiseless row.}", r"\end{table}"]
    (HERE / "sarry_scalib_joint_grid_table_v2.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))
    print("\nCHANGED CELLS (old table -> corrected table):" if changes else "\nNo cell changed at two decimals.")
    for c in changes:
        print("  " + c)

    # ---- item 2: NaN accounting of the ablation (sarry_scalib_diag2_step3_trials.jsonl)
    done = {}
    for line in (HERE / "sarry_scalib_diag2_step3_trials.jsonl").read_text().splitlines():
        r = json.loads(line)
        done[(r["version"], r["sigma"], r["k"], r["n_iters"])] = r
    rng = np.random.default_rng(12345)
    names = ["mean_rank", "top1_pct", "top5_pct", "top10_pct"]
    abl = {}
    print("\nABLATION, NaN accounting (differences = graph - HW-only, on the valid subset only)")
    print("| Version | sigma | Iterations | Attempted | Valid | Failure rate | HW-only mean rank (valid subset) | Graph mean rank | "
          "Diff mean rank [95% CI] | Diff top-1 % | Diff top-5 % | Diff top-10 % |")
    print("|" + "---|" * 12)
    for v in ("A", "B"):
        for s in (0.10, 0.50):
            for it in (10, 30, 100):
                rs = [done[(v, s, k, it)] for k in range(200)]
                ok = [r for r in rs if not r["nan"]]
                e = {"attempted": len(rs), "valid": len(ok), "failure_rate_pct": 100 * (len(rs) - len(ok)) / len(rs)}
                if ok:
                    g = np.array([d2.per_key(r["joint_bt"]) for r in ok]); h = np.array([d2.per_key(r["hw_only_bt"]) for r in ok])
                    mean_d, ci = d2.boot_ci(g - h, rng)
                    e.update(hw_only_on_valid_subset=dict(zip(names, h.mean(axis=0).tolist())),
                             graph_on_valid_subset=dict(zip(names, g.mean(axis=0).tolist())),
                             paired_diff={n_: {"mean": mean_d[i], "ci95": ci[i]} for i, n_ in enumerate(names)})
                    cells = [f"{h.mean(axis=0)[0]:.2f}", f"{g.mean(axis=0)[0]:.2f}"] + \
                            [f"{mean_d[i]:+.2f} [{ci[i][0]:+.2f}, {ci[i][1]:+.2f}]" for i in range(4)]
                else:
                    cells = ["-", "-", "not computable (no valid trial)", "-", "-", "-"]
                abl[f"{v}_{s}_{it}"] = e
                print(f"| {v} | {s:.2f} | {it} | {len(rs)} | {len(ok)} | {e['failure_rate_pct']:.1f}% | " + " | ".join(cells) + " |")
    (HERE / "sarry_scalib_final_checks_report.json").write_text(json.dumps(
        {"tie_convention": d2.CONVENTION, "std_definition": "standard deviation of the per-byte expected ranks",
         "rerun_identical_to_first_grid_trials": [same, len(rows)], "corrected_table": summ, "changed_cells": changes,
         "reference_hw_only": ref_hw, "ablation_nan_accounting": abl,
         "bootstrap": "10000 resamples over keys, percentile 95% CI, rng seed 12345"}, indent=1))
    print("REPORT DONE")


if __name__ == "__main__":
    {"sat": sat, "ties": ties_run, "report": report}[sys.argv[1]]()
