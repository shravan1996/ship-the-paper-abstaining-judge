"""Risk-controlled thresholds for an abstaining judge, after Badshah, Emami and Sajjad (2026).

Notation follows the paper. Each calibration item has a Mode 1 (no evidence) uncertainty u1 and
failure w1, and a Mode 2 (with retrieved evidence) uncertainty u2 and failure w2, where a failure
is a verdict that disagrees with the label. A verdict is accepted when its uncertainty is at or
below the mode's threshold. The risk R(t) is the share of accepted verdicts that are wrong, and
the target is Pr(R(t) <= alpha) >= 1 - delta over the draw of the calibration set (Eq. 4).

The Clopper-Pearson upper bound in Eq. 7 is BetaInv(1 - delta; w + 1, m - w). It is at or below
alpha exactly when the binomial CDF F(w; m, alpha) is at or below delta, which is the form used
here: it turns every check in the grid search into one table lookup.
"""

from __future__ import annotations

import math
from functools import lru_cache

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------------------------
# Clopper-Pearson
# ---------------------------------------------------------------------------------------------

def _log_pmf(k: int, m: int, p: float) -> float:
    return (math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)
            + k * math.log(p) + (m - k) * math.log1p(-p))


def binom_cdf(w: int, m: int, p: float) -> float:
    if w < 0:
        return 0.0
    if w >= m:
        return 1.0
    return min(1.0, sum(math.exp(_log_pmf(k, m, p)) for k in range(w + 1)))


def ucb(w: int, m: int, delta: float) -> float:
    """Clopper-Pearson one-sided upper bound on the error rate, by bisection on the binomial CDF."""
    if m == 0:
        return 1.0
    if w >= m:
        return 1.0
    lo, hi = w / m, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(w, m, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


@lru_cache(maxsize=None)
def wmax_table(alpha: float, delta: float, max_m: int) -> tuple[int, ...]:
    """wmax[m] is the most errors among m accepted verdicts that still certifies risk <= alpha.

    -1 means no error count certifies it, which is always the case for small m.
    """
    out = [-1] * (max_m + 1)
    log_a, log_1a = math.log(alpha), math.log1p(-alpha)
    for m in range(1, max_m + 1):
        cdf, w = 0.0, -1
        base = math.lgamma(m + 1)
        for k in range(m):
            cdf += math.exp(base - math.lgamma(k + 1) - math.lgamma(m - k + 1) + k * log_a + (m - k) * log_1a)
            if cdf > delta:
                break
            w = k
        out[m] = w
    return tuple(out)


def certified(m: int, w: int, alpha: float, delta: float, table: tuple[int, ...] | None = None) -> bool:
    if m <= 0:
        return False
    table = table or wmax_table(alpha, delta, m)
    return w <= table[m]


# ---------------------------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------------------------

def single_threshold(u: list[float], w: list[int], alpha: float, delta: float) -> float:
    """Eq. 8: the largest t whose every smaller candidate threshold also certifies risk <= alpha.

    Candidates are the sorted unique uncertainties, scanned upwards, stopping at the first one
    that fails. Returns -inf when the smallest candidate already fails, so nothing is accepted.
    """
    table = wmax_table(alpha, delta, len(u))
    order = sorted(range(len(u)), key=lambda i: u[i])
    best, m, errs, i = NEG_INF, 0, 0, 0
    while i < len(order):
        t = u[order[i]]
        while i < len(order) and u[order[i]] == t:
            m += 1
            errs += w[order[i]]
            i += 1
        if errs <= table[m]:
            best = t
        else:
            break
    return best


def joint_thresholds(u1: list[float], w1: list[int], u2: list[float], w2: list[int],
                     alpha: float, delta: float, conservative: bool = False) -> tuple[float, float]:
    """Algorithm 1 and Eq. 14: the (t1, t2) pair that accepts the most calibration items while
    its own Clopper-Pearson bound certifies risk <= alpha.

    Candidates are the sorted unique values of u1 and u2. The scan order (t1 ascending, then t2
    ascending, with a strict improvement rule) sets the tie-break exactly as in Algorithm 1.
    `conservative=True` runs every check at delta / (|T1| |T2|), the grid-corrected variant of
    Appendix D.4. The pointwise version, as published, makes no correction for the search.
    """
    n = len(u1)
    t1s = sorted(set(u1))
    t2s = sorted(set(u2))
    d = delta / (len(t1s) * len(t2s)) if conservative else delta
    table = wmax_table(alpha, d, n)

    rank2 = {t: j for j, t in enumerate(t2s)}
    cnt = [0] * len(t2s)   # items still routed to Mode 2, by their u2 rank
    err = [0] * len(t2s)
    for i in range(n):
        j = rank2[u2[i]]
        cnt[j] += 1
        err[j] += w2[i]
    by_u1 = sorted(range(n), key=lambda i: u1[i])

    best_m, best = 0, (NEG_INF, NEG_INF)
    m1 = e1 = 0
    pos = 0
    for t1 in t1s:
        while pos < n and u1[by_u1[pos]] == t1:
            i = by_u1[pos]
            m1 += 1
            e1 += w1[i]
            j = rank2[u2[i]]
            cnt[j] -= 1
            err[j] -= w2[i]
            pos += 1
        remaining = n - m1
        if m1 + remaining <= best_m:
            continue
        # Cumulative Mode 2 counts and errors at each t2.
        cc, ce = [0] * len(t2s), [0] * len(t2s)
        c = e = 0
        for j in range(len(t2s)):
            c += cnt[j]
            e += err[j]
            cc[j], ce[j] = c, e
        # m is non-decreasing in t2, so the largest feasible t2 gives this t1's best m.
        j = len(t2s) - 1
        while j >= 0:
            m = m1 + cc[j]
            if m <= best_m:
                break
            if e1 + ce[j] <= table[m]:
                # Smallest t2 with the same accepted set, matching Algorithm 1's first-hit rule.
                while j > 0 and cc[j - 1] == cc[j]:
                    j -= 1
                best_m, best = m, (t1, t2s[j])
                break
            j -= 1
    return best


def joint_split_certified(u1, w1, u2, w2, alpha: float, delta: float,
                          rungs: int = 9) -> tuple[float, float]:
    """Exploratory, not from the paper: pick candidates on one half, certify on the other.

    On the first half, Algorithm 1 is run at a ladder of targets from 0.2 alpha up to alpha,
    giving pairs in order of increasing aggressiveness. On the second half, those pairs are tested
    in that fixed order with one Clopper-Pearson check each at level delta, stopping at the first
    failure (fixed-sequence testing, as in Learn then Test). The search no longer spends the
    guarantee, because the half that certifies never saw the search.
    """
    n = len(u1)
    a = list(range(0, n, 2))
    b = list(range(1, n, 2))
    pick = lambda xs, idx: [xs[i] for i in idx]
    ladder = []
    for r in range(rungs):
        target = alpha * (0.2 + 0.8 * r / (rungs - 1))
        pair = joint_thresholds(pick(u1, a), pick(w1, a), pick(u2, a), pick(w2, a), target, delta)
        if pair not in ladder:
            ladder.append(pair)
    table = wmax_table(alpha, delta, len(b))
    chosen = (NEG_INF, NEG_INF)
    for t1, t2 in ladder:
        m = e = 0
        for i in b:
            if u1[i] <= t1:
                m += 1
                e += w1[i]
            elif u2[i] <= t2:
                m += 1
                e += w2[i]
        if m > 0 and e <= table[m]:
            chosen = (t1, t2)
        else:
            break
    return chosen


# ---------------------------------------------------------------------------------------------
# Routing and evaluation
# ---------------------------------------------------------------------------------------------

def route(u1: float, u2: float, t1: float, t2: float) -> str:
    """Algorithm 2: accept Mode 1, else retrieve and accept Mode 2, else abstain."""
    if u1 <= t1:
        return "mode1"
    if u2 <= t2:
        return "mode2"
    return "abstain"


def evaluate(rows: list[dict], t1: float, t2: float) -> dict:
    """Risk, coverage and retrieval share of a threshold pair on a fixed set of rows."""
    acc = err = retrieved = m1 = 0
    for r in rows:
        if r["u1"] <= t1:
            acc += 1
            m1 += 1
            err += r["w1"]
        else:
            retrieved += 1
            if r["u2"] <= t2:
                acc += 1
                err += r["w2"]
    n = len(rows)
    return {"risk": err / acc if acc else 0.0, "coverage": acc / n, "retrieval": retrieved / n,
            "mode1_share": m1 / n, "accepted": acc, "errors": err}
