"""Run the examinee and both judge modes on every item, and record what the calibration needs.

Three passes, so each one reuses its system prompt from the server's prompt cache:

  answers   Qwen3.5 9B answers each question; that answer is the candidate.
  mode1     the same model judges the candidate with no evidence.
  mode2     BM25 retrieves three documents for the question, and the model judges again with
            them appended.

Algorithm 1 precomputes both modes for every calibration item, so both are recorded for every
item. The label is the exact-match admission function against the table's answer. Every pass is
resumable: rows already on disk are skipped. Needs a local Ollama serving qwen3.5:9b; point
OLLAMA_CHAT_URL at it if it is not on the default port.

    python3 collect.py              # all three passes, then merge into data/judgments.jsonl
    python3 collect.py --workers 4  # concurrent requests; match OLLAMA_NUM_PARALLEL
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from judge import model
from judge.facts import admit, gold_title
from judge.retrieve import BM25, format_results

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def load(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_pass(name, items, fn, workers):
    out = DATA / f"pass_{name}.jsonl"
    done = {r["id"] for r in load(out)}
    todo = [it for it in items if it["id"] not in done]
    print(f"[{name}] {len(done)} done, {len(todo)} to go", flush=True)
    lock = threading.Lock()
    start, count = time.time(), [0]
    with open(out, "a") as f, ThreadPoolExecutor(workers) as pool:
        def work(it):
            row = {"id": it["id"], **fn(it)}
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
                count[0] += 1
                if count[0] % 100 == 0:
                    rate = (time.time() - start) / count[0]
                    print(f"[{name}] {count[0]}/{len(todo)}  {rate:.2f}s per item  ~{rate * (len(todo) - count[0]) / 60:.1f} min left", flush=True)
        list(pool.map(work, todo))
    return {r["id"]: r for r in load(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--nogold", action="store_true",
                    help="exploratory: a fourth pass where the answering document is removed from the results")
    ap.add_argument("--nogold-max-q", type=float, default=1.0,
                    help="only judge items explore.py could degrade at this share or below")
    args = ap.parse_args()

    items = load(DATA / "items.jsonl")
    index = BM25(load(DATA / "corpus.jsonl"))

    answers = run_pass("answers", items, lambda it: model.answer(it["q"]), args.workers)

    def mode1(it):
        return model.judge(model.MODE1_SYSTEM, model.mode1_user(it["q"], answers[it["id"]]["candidate"]))

    def mode2(it):
        t0 = time.perf_counter()
        hits = index.search(it["q"], k=3)
        retrieval_s = time.perf_counter() - t0
        j = model.judge(model.MODE2_SYSTEM, model.mode2_user(it["q"], answers[it["id"]]["candidate"], format_results(hits)))
        return {**j, "evidence_ids": [h["id"] for h in hits], "retrieval_s": retrieval_s}

    m1 = run_pass("mode1", items, mode1, args.workers)
    m2 = run_pass("mode2", items, mode2, args.workers)

    if args.nogold:
        # Added after the pre-registered run. The corpus holds the answer to every question, so
        # Mode 2 was right 99.3% of the time. This pass asks what the judge does when search comes
        # back with neighbours of the right row and not the row itself, which is how the web fails
        # (the paper's Appendix E.3 lists off-topic retrieval and insufficient evidence).
        def mode2_nogold(it):
            gold = gold_title(it)
            hits = [h for h in index.search(it["q"], k=6) if h["title"] != gold][:3]
            j = model.judge(model.MODE2_SYSTEM, model.mode2_user(it["q"], answers[it["id"]]["candidate"], format_results(hits)))
            return {**j, "evidence_ids": [h["id"] for h in hits]}
        # explore.py degrades an item at share q when this fixed draw is below q, so items whose
        # draw is above the largest q analysed are never needed.
        degradable = [it for it in items if random.Random(f"nogold:{it['id']}").random() < args.nogold_max_q]
        nogold = run_pass("mode2_nogold", degradable, mode2_nogold, args.workers)
        written = 0
        with open(DATA / "judgments_nogold.jsonl", "w") as f:
            for it in items:
                if it["id"] not in nogold:
                    continue
                written += 1
                j = nogold[it["id"]]
                f.write(json.dumps({"id": it["id"], "v2": j["v"], "p2": j["p_true"], "u2": j["u"],
                                    "parsed2": j["parsed"], "evidence_ids": j["evidence_ids"],
                                    "mode2_s": j["seconds"], "mode2_prompt_tokens": j["prompt_tokens"]}) + "\n")
        print(f"wrote {written} rows to data/judgments_nogold.jsonl", flush=True)

    with open(DATA / "judgments.jsonl", "w") as f:
        for it in items:
            a, j1, j2 = answers[it["id"]], m1[it["id"]], m2[it["id"]]
            f.write(json.dumps({
                "id": it["id"], "family": it["family"], "slice": it["slice"], "rare": it["rare"],
                "candidate": a["candidate"], "reference": it["display"], "e": admit(it, a["candidate"]),
                "v1": j1["v"], "p1": j1["p_true"], "u1": j1["u"], "parsed1": j1["parsed"],
                "v2": j2["v"], "p2": j2["p_true"], "u2": j2["u"], "parsed2": j2["parsed"],
                "evidence_ids": j2["evidence_ids"],
                "answer_s": a["seconds"], "mode1_s": j1["seconds"], "mode2_s": j2["seconds"],
                "retrieval_s": j2["retrieval_s"],
                "mode1_prompt_tokens": j1["prompt_tokens"], "mode2_prompt_tokens": j2["prompt_tokens"],
            }) + "\n")
    print(f"wrote {len(items)} rows to data/judgments.jsonl", flush=True)


if __name__ == "__main__":
    main()
