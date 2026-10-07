"""A small BM25 index over the reference corpus, standing in for the paper's web search.

The paper sends the question to the Serper API and appends the top 3 results, each with a title,
a snippet and a URL (Section 4, Appendix B.2). Here every family's reference table is one shared
corpus of short documents, and the question goes to BM25 unchanged. The corpus contains the right
answer for every question, so this retriever is kinder than the web: when it fails, it fails by
ranking a near neighbour above the right row, never by the answer being absent.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

TOKEN = re.compile(r"[a-z0-9]+")
STOP = {"the", "of", "a", "an", "is", "what", "which", "in", "to", "on", "for", "and", "was",
        "named", "number", "form", "answer", "or"}


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[dict], k1: float = 1.2, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf: list[Counter] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(docs):
            tf = Counter(tokens(d["title"] + " " + d["text"]))
            self.tf.append(tf)
            for t in tf:
                self.postings[t].append(i)
        self.len = [sum(tf.values()) for tf in self.tf]
        self.avg = sum(self.len) / len(self.len)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query: str, k: int = 3) -> list[dict]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(tokens(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [self.docs[i] for i, _ in best]


def format_results(results: list[dict]) -> str:
    """The paper's layout: an index, a title, a snippet and a source per result."""
    lines = []
    for i, d in enumerate(results, 1):
        lines.append(f"[{i}] Title: {d['title']}\n    Snippet: {d['text']}\n    Source: {d['source']}")
    return "\n".join(lines) if lines else "(no results)"
