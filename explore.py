"""Exploratory analyses, added after the pre-registered run. None of these is part of the decision.

The pre-registered run passed for a reason that made it uninformative: the reference corpus holds
the answer to every question, so the judge with evidence was right on 99.3% of items, and the
paper's tie-break sent 99.9% of verdicts to retrieval. Two changes, each a question I could not
answer from that run:

1. Cost-aware tie-break. Keep Eq. 14's objective, but among pairs that accept the same number of
   verdicts, take the one that retrieves least. Does saving retrieval spend the guarantee?
2. Search that misses. For a fixed share q of items, Mode 2 sees the three nearest neighbours of
   the answering document and not the document itself (data/judgments_nogold.jsonl). Each item
   gets one fixed uniform draw, so the items degraded at q = 0.25 are a subset of those at 0.5.
   Does the guarantee hold once the judge with evidence can be fooled?

    python3 explore.py              # writes results/exploratory.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path

import eval as E
from judge.calibrate import NEG_INF, evaluate, joint_thresholds, single_threshold

ROOT = Path(__file__).parent
DELTA = E.DELTA

E.METHODS.update({
    "mode2_only_largest": lambda c, a: (NEG_INF, single_threshold(c["u2"], c["w2"], a, DELTA, rule="largest")),
    "joint_fewer_retrievals": lambda c, a: joint_thresholds(c["u1"], c["w1"], c["u2"], c["w2"], a, DELTA, fewer_retrievals=True),
})

ARMS = ["mode1_only_largest", "mode2_only_largest", "joint", "joint_fewer_retrievals",
        "joint_conservative", "joint_split"]
QS = [0.0, 0.25, 0.5]


def degraded(rows: list[dict], q: float) -> list[dict]:
    if q == 0:
        return [dict(r) for r in rows]
    nogold = {}
    with open(ROOT / "data" / "judgments_nogold.jsonl") as f:
        for line in f:
            r = json.loads(line)
            nogold[r["id"]] = r
    out = []
    for r in rows:
        r = dict(r)
        if random.Random(f"nogold:{r['id']}").random() < q:
            g = nogold[r["id"]]  # collect.py --nogold-max-q guarantees every degradable item is present
            r["u2"], r["v2"], r["p2"] = g["u2"], g["v2"], g["p2"]
            r["w2"] = int(r["v2"] != r["e"])
            r["degraded"] = True
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--qs", type=float, nargs="*", default=QS)
    ap.add_argument("--out", default="exploratory.json")
    args = ap.parse_args()

    base = E.load_rows()
    summary = {"draws": args.draws, "delta": DELTA, "alphas": E.ALPHAS, "by_q": {}}
    for q in args.qs:
        rows = degraded(base, q)
        everyone = list(range(len(rows)))
        fam_ids = {f: [i for i, r in enumerate(rows) if r["family"] == f] for f in E.FAMILIES}
        tasks = [("all", 70000 + d, everyone, everyone, 1000, ARMS) for d in range(args.draws)]
        shift_arms = ["mode1_only_largest", "joint", "joint_fewer_retrievals"]
        for k, f in enumerate(E.FAMILIES):
            others = [i for i in everyone if rows[i]["family"] != f]
            for d in range(args.draws):
                # Same seed for the paired conditions: one calibration draw, scored on the
                # family it never saw and on the families it came from.
                tasks.append((f"unseen_{f}", 80000 + 1000 * k + d, others, fam_ids[f], 1000, shift_arms))
                tasks.append((f"seen_others_{f}", 80000 + 1000 * k + d, others, others, 1000, shift_arms + ["joint_split"]))
                for n_new in (100, 250):
                    tasks.append((f"recal{n_new}_{f}", 90000 + 1000 * k + 10 * n_new + d, fam_ids[f], fam_ids[f], n_new, shift_arms + ["joint_split"]))
        with Pool(args.procs, initializer=E._init, initargs=(rows,)) as pool:
            records = [r for chunk in pool.imap_unordered(E._draw, tasks, chunksize=4) for r in chunk]
        mode2_acc = 1 - sum(r["w2"] for r in rows) / len(rows)
        summary["by_q"][f"{q:.2f}"] = {
            "mode2_accuracy": mode2_acc,
            "mode2_accuracy_by_family": {f: 1 - sum(rows[i]["w2"] for i in ids) / len(ids) for f, ids in fam_ids.items()},
            "conditions": E.summarise(records),
            "paper_style": paper_style(rows),
        }
        c = summary["by_q"][f"{q:.2f}"]["conditions"]["all"]
        print(f"\nq = {q:.2f}  mode2 accuracy {mode2_acc:.3f}")
        print("  arm                       alpha  viol   mean  p95    cover  retr")
        for m in ARMS:
            for a in ("0.05", "0.10", "0.20"):
                v = c[m][a]
                print(f"  {m:24s} {a}  {100 * v['violation_rate']:5.1f}%  {v['mean_risk']:.3f}  {v['p95_risk']:.3f}  "
                      f"{100 * v['mean_coverage']:5.1f}%  {100 * v['mean_retrieval']:5.1f}%")
        for f in E.FAMILIES:
            for m in ("mode1_only_largest", "joint", "joint_fewer_retrievals"):
                v = summary["by_q"][f"{q:.2f}"]["conditions"][f"unseen_{f}"][m]["0.10"]
                print(f"  unseen {f:10s} {m:24s} a=0.10 viol {100 * v['violation_rate']:5.1f}%  mean {v['mean_risk']:.3f}  cover {100 * v['mean_coverage']:5.1f}%  retr {100 * v['mean_retrieval']:5.1f}%")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / args.out).write_text(json.dumps(summary, indent=1) + "\n")


def paper_style(rows, splits=100, seed=13):
    """Figure 1's protocol on the same rows: 50/50 splits, mean and std of empirical test FDR."""
    rng = random.Random(seed)
    acc = {}
    for _ in range(splits):
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        cal = [rows[i] for i in idx[: len(idx) // 2]]
        test = [rows[i] for i in idx[len(idx) // 2:]]
        cols = E.columns(cal)
        for a in E.ALPHAS:
            for m in ("joint", "joint_fewer_retrievals"):
                r = evaluate(test, *E.METHODS[m](cols, a))
                acc.setdefault((m, a), []).append(r["risk"])
    import statistics
    return {m: {f"{a:.2f}": {"mean_test_fdr": statistics.fmean(v), "std_test_fdr": statistics.pstdev(v),
                             "share_of_splits_above_alpha": sum(1 for x in v if x > a) / len(v)}
                for (mm, a), v in acc.items() if mm == m}
            for m in ("joint", "joint_fewer_retrievals")}


if __name__ == "__main__":
    main()
