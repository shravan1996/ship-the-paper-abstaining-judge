"""The examinee and the judge, both Qwen3.5 9B served by a local Ollama.

The paper's judge emits "Decision: True/False" and an explanation, and its uncertainty is the
binary predictive entropy of the True and False tokens at the verdict position (Section 4,
Appendix B.2). Ollama returns top log-probabilities for every generated token, so the judge is
asked for the paper's format, and the verdict position is the first generated token that reads
True or False.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request

OLLAMA = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
MODEL = "qwen3.5:9b"
SEED = 7

EXAMINEE_SYSTEM = (
    "You answer factual questions. Reply with the answer only: no explanation and no full sentence."
)

MODE1_SYSTEM = """You are an impartial judge. You will be given a question and a candidate answer. Decide whether the candidate answer is correct, using only your own knowledge.

Respond in exactly this format:
Decision: True or False
Explanation: one short sentence

Example
Question: What is the capital of France?
Candidate Answer: Paris
Decision: True
Explanation: Paris is the capital of France."""

MODE2_SYSTEM = """You are an impartial judge. You will be given a question, a candidate answer, and search results. Decide whether the candidate answer is correct. Judge it against the search results: answer True if they support the candidate answer and False if they contradict it. If the search results are inconclusive, use your own judgment.

Respond in exactly this format:
Decision: True or False
Explanation: one short sentence

Example
Question: What is the capital of France?
Candidate Answer: Paris
Search Results:
[1] Title: Paris
    Snippet: Paris is the capital and largest city of France.
    Source: encyclopedia
Decision: True
Explanation: The first result states that Paris is the capital of France."""


def chat(system: str, user: str, num_predict: int, logprobs: bool) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "think": False,
        "stream": False,
        "options": {"temperature": 0, "seed": SEED, "num_predict": num_predict},
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 20
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))


def answer(question: str) -> dict:
    out = chat(EXAMINEE_SYSTEM, question, num_predict=24, logprobs=False)
    text = out["message"]["content"].strip().splitlines()
    return {"candidate": text[0].strip() if text else "", "seconds": out["total_duration"] / 1e9}


def binary_entropy(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def verdict_from_logprobs(steps: list[dict]) -> tuple[int, float, float, bool]:
    """(verdict, p_true, entropy, found) from the first generated True/False token.

    p_true renormalises the probability mass on True-like and False-like tokens in the top 20 at
    that position. The paper does not say whether it renormalises; without it, mass on " True"
    versus "True" would leak out of the entropy.
    """
    for step in steps:
        if step["token"].strip().lower() not in ("true", "false"):
            continue
        pt = pf = 0.0
        for alt in step["top_logprobs"]:
            word = alt["token"].strip().lower()
            if word == "true":
                pt += math.exp(alt["logprob"])
            elif word == "false":
                pf += math.exp(alt["logprob"])
        p_true = pt / (pt + pf) if pt + pf > 0 else 0.5
        return int(p_true >= 0.5), p_true, binary_entropy(p_true), True
    return 0, 0.5, math.log(2), False


def judge(system: str, user: str) -> dict:
    out = chat(system, user, num_predict=6, logprobs=True)
    v, p, u, found = verdict_from_logprobs(out.get("logprobs") or [])
    return {"v": v, "p_true": p, "u": u, "parsed": found, "seconds": out["total_duration"] / 1e9,
            "prompt_tokens": out.get("prompt_eval_count", 0)}


def mode1_user(q: str, a: str) -> str:
    return f"Question: {q}\nCandidate Answer: {a}"


def mode2_user(q: str, a: str, evidence: str) -> str:
    return f"Question: {q}\nCandidate Answer: {a}\nSearch Results:\n{evidence}"
