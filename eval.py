"""Every number in the README and the article, computed from data/judgments.jsonl.

No model is called here. The population is the 1,972 recorded items. A calibration set is drawn
i.i.d. with replacement from it, thresholds are chosen on that draw alone, and their risk and
coverage are then computed exactly on the population, which is the distribution the draw came
from. "Violation" means the population risk of the chosen thresholds is above alpha: the event
Eq. 4 says happens with probability at most delta.

    python3 eval.py                 # everything, about 25 minutes on 12 cores
    python3 eval.py --quick         # 100 draws per condition, for a smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from judge.calibrate import (NEG_INF, evaluate, joint_split_certified, joint_thresholds,
                             single_threshold)
from judge.facts import ELEMENTS

ROOT = Path(__file__).parent
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25]
DELTA = 0.05
FAMILIES = ["elements", "ports", "unicode", "timezones"]

METHODS = {
    "mode1_only": lambda c, a: (single_threshold(c["u1"], c["w1"], a, DELTA), NEG_INF),
    "mode1_only_largest": lambda c, a: (single_threshold(c["u1"], c["w1"], a, DELTA, rule="largest"), NEG_INF),
    "mode2_only": lambda c, a: (NEG_INF, single_threshold(c["u2"], c["w2"], a, DELTA)),
    "joint": lambda c, a: joint_thresholds(c["u1"], c["w1"], c["u2"], c["w2"], a, DELTA),
    "joint_conservative": lambda c, a: joint_thresholds(c["u1"], c["w1"], c["u2"], c["w2"], a, DELTA, conservative=True),
    "joint_split": lambda c, a: joint_split_certified(c["u1"], c["w1"], c["u2"], c["w2"], a, DELTA),
}


def load_rows() -> list[dict]:
    rows = []
    with open(ROOT / "data" / "judgments.jsonl") as f:
        for line in f:
            r = json.loads(line)
            r["w1"] = int(r["v1"] != r["e"])
            r["w2"] = int(r["v2"] != r["e"])
            rows.append(r)
    return rows


def columns(sample: list[dict]) -> dict:
    return {k: [r[k] for r in sample] for k in ("u1", "w1", "u2", "w2")}


# ---------------------------------------------------------------------------------------------
# Draw workers. Globals are set once per process by the pool initialiser.
# ---------------------------------------------------------------------------------------------

_POP: list[dict] = []


def _init(rows):
    global _POP
    _POP = rows


def _draw(task):
    """One calibration draw: (condition, seed, cal_ids, eval_ids, n, methods) -> per-alpha results."""
    cond, seed, cal_pool, eval_ids, n, methods = task
    rng = random.Random(seed)
    cal = [_POP[cal_pool[rng.randrange(len(cal_pool))]] for _ in range(n)]
    target = [_POP[i] for i in eval_ids]
    cols = columns(cal)
    out = []
    for a in ALPHAS:
        for m in methods:
            t1, t2 = METHODS[m](cols, a)
            res = evaluate(target, t1, t2)
            out.append((cond, m, a, res["risk"], res["coverage"], res["retrieval"], res["mode1_share"]))
    return out


def summarise(records) -> dict:
    groups = defaultdict(list)
    for cond, m, a, risk, cov, ret, m1 in records:
        groups[(cond, m, a)].append((risk, cov, ret, m1))
    out = {}
    for (cond, m, a), vals in groups.items():
        risks = sorted(v[0] for v in vals)
        n = len(vals)
        viol = sum(1 for r in risks if r > a + 1e-12)
        out.setdefault(cond, {}).setdefault(m, {})[f"{a:.2f}"] = {
            "draws": n,
            "violation_rate": viol / n,
            "violation_se": math.sqrt(max(viol / n * (1 - viol / n), 1e-12) / n),
            "mean_risk": statistics.fmean(risks),
            "p95_risk": risks[min(n - 1, int(0.95 * n))],
            "max_risk": risks[-1],
            "mean_coverage": statistics.fmean(v[1] for v in vals),
            "mean_retrieval": statistics.fmean(v[2] for v in vals),
            "mean_mode1_share": statistics.fmean(v[3] for v in vals),
            "zero_coverage_share": sum(1 for v in vals if v[1] == 0) / n,
        }
    return out


# ---------------------------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------------------------

def gold_title(row: dict, items: dict) -> str | None:
    q = items[row["id"]]["q"]
    if row["family"] == "elements":
        by_num = {str(z): n for z, _, n in ELEMENTS}
        by_sym = {s: n for _, s, n in ELEMENTS}
        m = re.search(r"atomic number (\d+)\?", q)
        if m:
            return f"Periodic table: {by_num[m.group(1)]}"
        m = re.search(r"symbol ([A-Z][a-z]?)\?", q)
        if m:
            return f"Periodic table: {by_sym[m.group(1)]}"
        m = re.search(r"(?:of|for) ([a-z]+)\?", q)
        return f"Periodic table: {m.group(1).capitalize()}" if m else None
    name = re.search(r'"([^"]+)"', q).group(1)
    return {"ports": f"IANA service registry: {name}", "unicode": f"Unicode character: {name}",
            "timezones": f"Time zone database: {name}"}[row["family"]]


def describe(rows: list[dict]) -> dict:
    items = {}
    with open(ROOT / "data" / "items.jsonl") as f:
        for line in f:
            it = json.loads(line)
            items[it["id"]] = it
    titles = {}
    with open(ROOT / "data" / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            titles[d["id"]] = d["title"]

    def block(rs):
        n = len(rs)
        m1_fail = [r for r in rs if r["w1"]]
        gold = [gold_title(r, items) in {titles[i] for i in r["evidence_ids"]} for r in rs]
        return {
            "items": n,
            "examinee_accuracy": sum(r["e"] for r in rs) / n,
            "mode1_accuracy": 1 - sum(r["w1"] for r in rs) / n,
            "mode2_accuracy": 1 - sum(r["w2"] for r in rs) / n,
            "mode1_says_true": sum(r["v1"] for r in rs) / n,
            "mode1_false_accept_share_of_failures": (sum(1 for r in m1_fail if r["v1"] == 1) / len(m1_fail)) if m1_fail else 0.0,
            "mode1_failures_with_u_below_0.05": sum(1 for r in m1_fail if r["u1"] < 0.05),
            "mode1_failures": len(m1_fail),
            "mode2_failures": sum(r["w2"] for r in rs),
            "retrieval_gold_in_top3": sum(gold) / n,
            "unparsed_verdicts": sum(1 for r in rs if not r["parsed1"]) + sum(1 for r in rs if not r["parsed2"]),
            "retrieval_raised_uncertainty": sum(1 for r in rs if r["u2"] > r["u1"]) / n,
        }

    min_u1 = min(r["u1"] for r in rows)
    out = {"all": block(rows), "by_family": {f: block([r for r in rows if r["family"] == f]) for f in FAMILIES}}
    out["all"]["min_u1_ties"] = sum(1 for r in rows if r["u1"] == min_u1)
    out["all"]["distinct_u1"] = len({r["u1"] for r in rows})
    out["all"]["distinct_u2"] = len({r["u2"] for r in rows})
    for k in ("answer_s", "mode1_s", "mode2_s", "retrieval_s", "mode1_prompt_tokens", "mode2_prompt_tokens"):
        out["all"][f"mean_{k}"] = statistics.fmean(r[k] for r in rows)
    confident_wrong = sorted((r for r in rows if r["w1"] and r["v1"] == 1), key=lambda r: r["u1"])[:8]
    out["most_confident_mode1_false_accepts"] = [
        {"id": r["id"], "q": items[r["id"]]["q"], "candidate": r["candidate"], "reference": r["reference"],
         "p_true": r["p1"], "u1": r["u1"], "mode2_verdict": r["v2"], "u2": r["u2"]} for r in confident_wrong]
    return out


def paper_style_splits(rows, splits=100, seed=11):
    """Figure 1's protocol: random 50/50 splits, empirical FDR on the test half, mean and std."""
    rng = random.Random(seed)
    res = defaultdict(list)
    for _ in range(splits):
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        cal = [rows[i] for i in idx[: len(idx) // 2]]
        test = [rows[i] for i in idx[len(idx) // 2:]]
        cols = columns(cal)
        for a in ALPHAS:
            for m in ("mode1_only", "joint"):
                r = evaluate(test, *METHODS[m](cols, a))
                res[(m, a)].append((r["risk"], r["coverage"]))
    out = {}
    for (m, a), v in res.items():
        risks = [x[0] for x in v]
        out.setdefault(m, {})[f"{a:.2f}"] = {
            "mean_test_fdr": statistics.fmean(risks), "std_test_fdr": statistics.pstdev(risks),
            "share_of_splits_above_alpha": sum(1 for r in risks if r > a) / len(risks),
            "mean_coverage": statistics.fmean(x[1] for x in v)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    primary_draws = 100 if args.quick else 1000
    secondary_draws = 100 if args.quick else 500

    rows = load_rows()
    everyone = list(range(len(rows)))
    fam_ids = {f: [i for i, r in enumerate(rows) if r["family"] == f] for f in FAMILIES}
    all_methods = list(METHODS)
    core = ["mode1_only", "mode1_only_largest", "joint", "joint_conservative", "joint_split"]

    tasks = []
    # A. Primary: n = 1,000 from the whole population.
    for d in range(primary_draws):
        tasks.append(("primary", 1000 + d, everyone, everyone, 1000, all_methods))
    # B. Calibration size.
    for n in (250, 500):
        for d in range(secondary_draws):
            tasks.append((f"size_{n}", 20000 + n * 10 + d, everyone, everyone, n, core))
    # C. A family the calibration never saw, and D. recalibration on that family alone.
    for k, f in enumerate(FAMILIES):
        others = [i for i in everyone if rows[i]["family"] != f]
        for d in range(secondary_draws):
            tasks.append((f"unseen_{f}", 40000 + 1000 * k + d, others, fam_ids[f], 1000, core))
            tasks.append((f"seen_others_{f}", 40000 + 1000 * k + d, others, others, 1000, ["joint"]))
            for n_new in (100, 250):
                tasks.append((f"recal{n_new}_{f}", 60000 + 1000 * k + 10 * n_new + d, fam_ids[f], fam_ids[f], n_new, core))

    with Pool(args.procs, initializer=_init, initargs=(rows,)) as pool:
        records = [r for chunk in pool.imap_unordered(_draw, tasks, chunksize=4) for r in chunk]

    summary = {
        "delta": DELTA,
        "alphas": ALPHAS,
        "primary_draws": primary_draws,
        "secondary_draws": secondary_draws,
        "describe": describe(rows),
        "paper_style": paper_style_splits(rows, splits=20 if args.quick else 100),
        "conditions": summarise(records),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    name = "summary_quick.json" if args.quick else "summary.json"
    (ROOT / "results" / name).write_text(json.dumps(summary, indent=1) + "\n")
    print_report(summary)


def pct(x):
    return f"{100 * x:.1f}%"


def print_report(s: dict) -> None:
    d = s["describe"]["all"]
    print(f"items {d['items']}  examinee accuracy {pct(d['examinee_accuracy'])}  "
          f"mode1 accuracy {pct(d['mode1_accuracy'])}  mode2 accuracy {pct(d['mode2_accuracy'])}")
    for f, b in s["describe"]["by_family"].items():
        print(f"  {f:10s} examinee {pct(b['examinee_accuracy'])}  m1 {pct(b['mode1_accuracy'])}  "
              f"m2 {pct(b['mode2_accuracy'])}  gold-in-top3 {pct(b['retrieval_gold_in_top3'])}")
    c = s["conditions"]
    print("\nprimary, n = 1,000:  method  alpha  violation  mean risk  p95 risk  coverage  retrieval")
    for m, by_a in c["primary"].items():
        for a, v in by_a.items():
            print(f"  {m:20s} {a}  {pct(v['violation_rate']):>6s}  {v['mean_risk']:.3f}  {v['p95_risk']:.3f}  "
                  f"{pct(v['mean_coverage']):>6s}  {pct(v['mean_retrieval']):>6s}")
    for f in FAMILIES:
        v = c[f"unseen_{f}"]["joint"]["0.10"]
        print(f"unseen {f:10s} joint a=0.10: violation {pct(v['violation_rate'])}  mean risk {v['mean_risk']:.3f}  coverage {pct(v['mean_coverage'])}")


if __name__ == "__main__":
    main()
