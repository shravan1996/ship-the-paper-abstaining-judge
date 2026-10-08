import unittest

from judge.facts import (ELEMENTS, admit, format_offset, norm_codepoint, norm_element,
                         norm_offset, norm_symbol)
from judge.model import binary_entropy, verdict_from_logprobs
from judge.retrieve import BM25


class Elements(unittest.TestCase):
    def test_table_is_complete_and_unique(self):
        self.assertEqual([z for z, _, _ in ELEMENTS], list(range(1, 119)))
        self.assertEqual(len({s for _, s, _ in ELEMENTS}), 118)
        self.assertEqual(len({n for _, _, n in ELEMENTS}), 118)

    def test_spelling_variants_and_symbols(self):
        self.assertEqual(norm_element("Aluminum"), "aluminium")
        self.assertEqual(norm_element("Cesium."), "caesium")
        self.assertEqual(norm_element("Tm"), "thulium")
        self.assertEqual(norm_symbol("Tm"), "tm")
        self.assertEqual(norm_symbol("The symbol is W"), "w")


class Normalisers(unittest.TestCase):
    def test_offsets(self):
        self.assertEqual(norm_offset("+05:45"), 345)
        self.assertEqual(norm_offset("UTC-3:30"), -210)
        self.assertEqual(norm_offset("GMT−03:00"), -180)
        self.assertEqual(norm_offset("UTC"), 0)
        self.assertEqual(norm_offset("UTC±00:00"), 0)
        self.assertIsNone(norm_offset("I am not sure"))
        self.assertEqual(format_offset(-210), "-03:30")

    def test_codepoints(self):
        self.assertEqual(norm_codepoint("U+03BB"), 0x3BB)
        self.assertEqual(norm_codepoint("U+3BB"), 0x3BB)
        self.assertEqual(norm_codepoint("0x2603"), 0x2603)
        self.assertIsNone(norm_codepoint("lambda"))

    def test_admission_is_exact(self):
        item = {"kind": "int", "ref": 993}
        self.assertEqual(admit(item, "993"), 1)
        self.assertEqual(admit(item, "Port 143"), 0)
        self.assertEqual(admit(item, ""), 0)


class Verdicts(unittest.TestCase):
    def test_entropy_bounds(self):
        self.assertAlmostEqual(binary_entropy(0.5), 0.6931, places=4)
        self.assertLess(binary_entropy(0.999), 0.01)

    def test_reads_first_true_false_token_and_renormalises(self):
        steps = [
            {"token": "Decision", "top_logprobs": [{"token": "Decision", "logprob": -0.01}]},
            {"token": ":", "top_logprobs": [{"token": ":", "logprob": 0.0}]},
            {"token": " True", "top_logprobs": [
                {"token": " True", "logprob": -0.2231},   # 0.8
                {"token": " False", "logprob": -1.8971},  # 0.15
                {"token": "True", "logprob": -2.9957},    # 0.05
            ]},
        ]
        v, p, u, found = verdict_from_logprobs(steps)
        self.assertTrue(found)
        self.assertEqual(v, 1)
        self.assertAlmostEqual(p, 0.85, places=2)

    def test_missing_verdict_is_maximally_uncertain(self):
        v, p, u, found = verdict_from_logprobs([{"token": "Maybe", "top_logprobs": []}])
        self.assertFalse(found)
        self.assertAlmostEqual(u, binary_entropy(0.5))


class Retrieval(unittest.TestCase):
    def test_exact_name_ranks_first(self):
        docs = [{"id": i, "title": t, "text": x, "source": "s"} for i, (t, x) in enumerate([
            ("Unicode character: GREEK SMALL LETTER LAMDA", "U+03BB GREEK SMALL LETTER LAMDA"),
            ("Unicode character: GREEK CAPITAL LETTER LAMDA", "U+039B GREEK CAPITAL LETTER LAMDA"),
            ("IANA service registry: imaps", "imaps 993/tcp"),
        ])]
        top = BM25(docs).search('code point of the character named "GREEK SMALL LETTER LAMDA"', k=2)
        self.assertEqual(top[0]["id"], 0)


if __name__ == "__main__":
    unittest.main()
