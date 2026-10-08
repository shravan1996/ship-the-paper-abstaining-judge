import itertools
import math
import random
import unittest

from judge.calibrate import (NEG_INF, binom_cdf, certified, evaluate, joint_split_certified,
                             joint_thresholds, route, single_threshold, ucb, wmax_table)


def brute_joint(u1, w1, u2, w2, alpha, delta):
    """Algorithm 1 written the slow, literal way: every pair, ascending, strict improvement."""
    best_m, best = 0, (NEG_INF, NEG_INF)
    for t1 in sorted(set(u1)):
        for t2 in sorted(set(u2)):
            m = w = 0
            for a, b, c, d in zip(u1, w1, u2, w2):
                if a <= t1:
                    m, w = m + 1, w + b
                elif c <= t2:
                    m, w = m + 1, w + d
            if m > 0 and ucb(w, m, delta) <= alpha and m > best_m:
                best_m, best = m, (t1, t2)
    return best


class ClopperPearson(unittest.TestCase):
    def test_zero_errors_matches_closed_form(self):
        # With no errors the one-sided bound is 1 - delta^(1/m).
        for m in (5, 10, 50):
            self.assertAlmostEqual(ucb(0, m, 0.05), 1 - 0.05 ** (1 / m), places=6)

    def test_table_agrees_with_bisection(self):
        rng = random.Random(0)
        for _ in range(300):
            m = rng.randint(1, 120)
            w = rng.randint(0, m)
            alpha = rng.choice([0.05, 0.1, 0.2])
            self.assertEqual(certified(m, w, alpha, 0.05), ucb(w, m, 0.05) <= alpha, (m, w, alpha))

    def test_small_samples_never_certify(self):
        # Ten clean verdicts cannot certify a 10% risk at 95% confidence: the bound is 0.259.
        self.assertFalse(certified(10, 0, 0.10, 0.05))
        # Zero errors certifies 10% once 0.9^m <= 0.05, which first happens at m = 29.
        self.assertEqual(wmax_table(0.10, 0.05, 30)[29], 0)
        self.assertEqual(wmax_table(0.10, 0.05, 30)[28], -1)

    def test_binom_cdf_edges(self):
        self.assertEqual(binom_cdf(-1, 10, 0.3), 0.0)
        self.assertEqual(binom_cdf(10, 10, 0.3), 1.0)
        self.assertAlmostEqual(binom_cdf(0, 3, 0.5), 0.125)


class Thresholds(unittest.TestCase):
    def test_single_threshold_stops_at_first_failure(self):
        # The three most confident verdicts are wrong, the next 200 right. A threshold above all
        # of them would certify, but Eq. 8 needs every smaller threshold to certify too.
        u = [0.01, 0.02, 0.03] + [0.1 + i / 1000 for i in range(200)]
        w = [1, 1, 1] + [0] * 200
        self.assertEqual(single_threshold(u, w, 0.10, 0.05), NEG_INF)

    def test_prefix_rule_needs_a_large_first_group(self):
        # All distinct: the first threshold holds one verdict and cannot certify, so Eq. 8 as
        # written accepts nothing, while the "largest certified threshold" reading accepts the
        # clean prefix.
        u = [i / 100 for i in range(100)]
        w = [0] * 60 + [1] * 40
        self.assertEqual(single_threshold(u, w, 0.10, 0.05), NEG_INF)
        t = single_threshold(u, w, 0.10, 0.05, rule="largest")
        self.assertGreater(t, 0.3)
        self.assertLessEqual(t, 0.65)
        # Forty tied at the most confident value: now the prefix rule has room to start.
        u_tied = [0.0] * 40 + [i / 100 for i in range(1, 61)]
        w_tied = [0] * 60 + [1] * 40
        self.assertGreater(single_threshold(u_tied, w_tied, 0.10, 0.05), 0.1)

    def test_joint_matches_literal_algorithm_one(self):
        rng = random.Random(1)
        for trial in range(25):
            n = rng.randint(15, 40)
            u1 = [round(rng.random(), 2) for _ in range(n)]
            u2 = [round(rng.random(), 2) for _ in range(n)]
            w1 = [int(rng.random() < 0.15 + 0.5 * x) for x in u1]
            w2 = [int(rng.random() < 0.02 + 0.3 * x) for x in u2]
            alpha = rng.choice([0.15, 0.25, 0.35])
            self.assertEqual(joint_thresholds(u1, w1, u2, w2, alpha, 0.05),
                             brute_joint(u1, w1, u2, w2, alpha, 0.05), trial)

    def test_conservative_never_accepts_more(self):
        rng = random.Random(2)
        u1 = [rng.random() for _ in range(300)]
        u2 = [rng.random() for _ in range(300)]
        w1 = [int(rng.random() < 0.4 * x) for x in u1]
        w2 = [int(rng.random() < 0.2 * x) for x in u2]
        rows = [{"u1": a, "w1": b, "u2": c, "w2": d} for a, b, c, d in zip(u1, w1, u2, w2)]
        loose = evaluate(rows, *joint_thresholds(u1, w1, u2, w2, 0.1, 0.05))
        tight = evaluate(rows, *joint_thresholds(u1, w1, u2, w2, 0.1, 0.05, conservative=True))
        self.assertLessEqual(tight["coverage"], loose["coverage"])

    def test_split_variant_certifies_on_held_out_half(self):
        rng = random.Random(3)
        u1 = [rng.random() for _ in range(400)]
        u2 = [rng.random() for _ in range(400)]
        w1 = [int(rng.random() < 0.3 * x) for x in u1]
        w2 = [int(rng.random() < 0.1 * x) for x in u2]
        t1, t2 = joint_split_certified(u1, w1, u2, w2, 0.1, 0.05)
        b = list(range(1, 400, 2))
        rows = [{"u1": u1[i], "w1": w1[i], "u2": u2[i], "w2": w2[i]} for i in b]
        res = evaluate(rows, t1, t2)
        if res["accepted"]:
            self.assertTrue(certified(res["accepted"], res["errors"], 0.1, 0.05))


class Routing(unittest.TestCase):
    def test_algorithm_two(self):
        self.assertEqual(route(0.1, 0.9, 0.2, 0.3), "mode1")
        self.assertEqual(route(0.5, 0.2, 0.2, 0.3), "mode2")
        self.assertEqual(route(0.5, 0.4, 0.2, 0.3), "abstain")
        self.assertEqual(route(0.0, 0.0, NEG_INF, NEG_INF), "abstain")

    def test_evaluate_counts_errors_by_mode(self):
        rows = [{"u1": 0.1, "w1": 1, "u2": 0.0, "w2": 0},
                {"u1": 0.9, "w1": 1, "u2": 0.1, "w2": 0},
                {"u1": 0.9, "w1": 0, "u2": 0.9, "w2": 1}]
        res = evaluate(rows, 0.5, 0.5)
        self.assertEqual((res["accepted"], res["errors"]), (2, 1))
        self.assertAlmostEqual(res["retrieval"], 2 / 3)


class GuaranteeOnSimulatedData(unittest.TestCase):
    def test_single_mode_holds_on_iid_draws(self):
        # A population whose error rate rises with uncertainty. Eq. 8 should exceed alpha in at
        # most delta of calibration draws, up to Monte Carlo error.
        rng = random.Random(4)
        pop = [(x, int(rng.random() < 0.25 * x)) for x in (rng.random() for _ in range(3000))]
        alpha, delta, draws, bad = 0.08, 0.05, 200, 0
        for _ in range(draws):
            cal = [pop[rng.randrange(len(pop))] for _ in range(400)]
            t = single_threshold([c[0] for c in cal], [c[1] for c in cal], alpha, delta)
            acc = [w for u, w in pop if u <= t]
            if acc and sum(acc) / len(acc) > alpha:
                bad += 1
        self.assertLessEqual(bad / draws, delta + 3 * math.sqrt(delta * (1 - delta) / draws))


if __name__ == "__main__":
    unittest.main()
