# ship-the-paper-abstaining-judge

A small test of one claim from **[Judge, Retrieve, or Abstain: Uncertainty-Guarded LLM Judging with Provable Risk Guarantees](https://arxiv.org/abs/2608.17994)** (Sher Badshah, Ali Emami and Hassan Sajjad, arXiv:2608.17994, first version 18 August 2026, accepted at COLM 2026).

Companion prototype for the *Ship the Paper* article on the abstaining judge.

## The research question

The paper's judge reads a question and a candidate answer and says True or False. Its uncertainty is the binary entropy of its own True/False probability at the verdict token. A verdict is accepted when that uncertainty is at or below a threshold; otherwise the judge retrieves three search results and decides again under a second threshold; otherwise it abstains and a person reviews the item. Both thresholds are calibrated on labelled verdicts with a Clopper-Pearson bound, and the paper's promise (Eq. 4) is that with probability at least 1 − δ over the calibration draw, at most α of the accepted verdicts are wrong.

The paper reports the mean and standard deviation of test-set error over 100 calibration/test splits (Figure 1). The promise is about the tail: how often a calibration draw produces thresholds whose true risk is above α. And Algorithm 1 picks the best of many threshold pairs, each checked at δ, while the proof (Lemma 1) is written for one fixed pair. So this prototype asks:

> Calibrated on 1,000 labelled verdicts, does the paper's two-threshold router keep the share of wrong verdicts among the verdicts it accepts at or below α in at least 1 − δ of calibration draws, while accepting at least 10 points more verdicts than the judge alone?

The thresholds were committed before any judgment was collected, in [PREREGISTRATION.md](PREREGISTRATION.md) (commit `3b26cb1`).

## Setup

```bash
git clone https://github.com/shravan1996/ship-the-paper-abstaining-judge
cd ship-the-paper-abstaining-judge
python3 -m unittest discover -s tests -t .
python3 eval.py                 # the pre-registered analysis, about 5 minutes on 10 cores
python3 explore.py              # the exploratory analyses, about 30 minutes on 10 cores
```

Python 3.11 or newer, standard library only. Neither script calls a model: they read `data/judgments.jsonl` and `data/judgments_nogold.jsonl` and write `results/summary.json` and `results/exploratory.json`.

To regenerate the judgments you need a local [Ollama](https://ollama.com) serving `qwen3.5:9b`:

```bash
python3 build_data.py           # data/items.jsonl and data/corpus.jsonl from the reference tables
OLLAMA_CHAT_URL=http://127.0.0.1:11434/api/chat python3 collect.py --workers 4
python3 collect.py --workers 4 --nogold --nogold-max-q 0.5   # the exploratory pass
```

The git history keeps the order honest: `3b26cb1` committed the pre-registration before any judgment existed, and `60fcbc3` recorded the pre-registered run before any exploratory analysis was run.

## What is in the data

Every reference answer comes from a table, so a label is never a model's opinion. The admission function is an exact match after normalisation (`judge/facts.py`).

| Family | Items | Source of the answer | Examinee right | Judge right, from memory | Judge right, with evidence |
| --- | ---: | --- | ---: | ---: | ---: |
| elements | 472 | periodic table in `judge/facts.py` | 99.6% | 97.2% | 99.2% |
| ports | 500 | IANA registry in macOS `/etc/services` | 4.0% | 40.0% | 99.8% |
| unicode | 500 | Python `unicodedata`, Unicode 16.0 | 10.8% | 23.6% | 98.8% |
| timezones | 500 | system tzdata 2026c, offsets on 15 Jan and 15 Jul 2025 | 80.4% | 78.6% | 99.4% |

- **Model.** `qwen3.5:9b` (Q4_K_M) in Ollama 0.34.4 on an Apple M4 Pro, both as the examinee and as the judge, temperature 0, thinking off, otherwise Ollama's defaults for the model.
- **Uncertainty.** Binary entropy of the True/False probability at the first generated True or False token, renormalised over the True-like and False-like tokens in the top 20 (`judge/model.py`).
- **Mode 2 evidence.** BM25 top 3 over 15,772 reference rows (`judge/retrieve.py`). The answering row is in the top 3 for 99.8% of questions, which is much kinder than web search.
- **Truth for the guarantee.** The 1,972 recorded items are the population. A calibration set is drawn from it i.i.d. with replacement, and the chosen thresholds are scored exactly on the whole population.

`data/judgments.jsonl` holds one row per item: candidate, label, both verdicts, both probabilities, both uncertainties, the retrieved row ids and timings. The timings were recorded on a GPU shared with another workload and are not a latency benchmark; prompt token counts (135 for Mode 1 and 364 for Mode 2, on average) are the stable cost measure.

## Pre-registered result: Continue

Primary analysis, 1,000 calibration draws of 1,000 items each, δ = 0.05.

| Arm | α | Draws over target | Mean risk | Kept | Looked it up |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mode1_only` (Eq. 8 as written) | 0.10 | 0.0% | n/a | 0.0% | none |
| `mode1_only_largest` | 0.05 | 4.4% | 2.9% | 27.5% | none |
| `mode1_only_largest` | 0.10 | 3.5% | 7.5% | 39.1% | none |
| `mode1_only_largest` | 0.20 | 4.1% | 16.9% | 49.9% | none |
| **`joint` (Algorithm 1)** | every α | **0.0%** | **0.7%** | **99.9%** | **99.9%** |
| `joint_conservative` | every α | 0.0% | 0.7% | 99.9% | 99.9% |
| `joint_split` (exploratory) | every α | 0.0% | 0.6% | 99.8% | 99.8% |

The decision rule fired **Continue**: no draws over target at α = 0.10, and a coverage gain of 99.9 points over `mode1_only`. The paper-style protocol (100 random 50/50 splits) gives `joint` a mean test error of 0.63% with a standard deviation of 0.19 points.

Three of the six predictions were wrong. The printed search did not overshoot (predicted 8% to 25%), the grid correction cost no coverage (predicted at least half the gain), and on a held-out family the printed router did not overshoot (predicted most draws). All three missed for one reason: with evidence, the judge was right on 99.3% of items, so Algorithm 1 could keep every item at a risk far below any target, and its tie-break (strict improvement while scanning t1 upwards) then chose the smallest t1. The router answered one item from memory and retrieved for the rest. `mode1_only` as written accepts nothing because the smallest of 1,972 distinct uncertainty values holds a single verdict, which cannot certify; `mode1_only_largest` reads Eq. 8 as "the largest threshold whose bound certifies", the sentence that introduces it.

## Exploratory results

Added after the pre-registered run, in `explore.py` (500 draws per condition). None of this is part of the decision.

**Cost-aware tie-break** (`joint_thresholds(..., fewer_retrievals=True)`): same objective and check, but among pairs that keep the same number of verdicts, take the largest t1.

| α | Draws over target | Mean risk | 95th percentile risk | Kept | Looked it up |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 4.0% | 3.8% | 4.9% | 99.9% | 59.5% |
| 0.10 | 4.0% | 8.4% | 9.9% | 99.9% | 50.8% |
| 0.20 | 2.6% | 17.7% | 19.8% | 99.9% | 36.4% |

**A family the calibration never saw** (calibrate on three families, score on the fourth, α = 0.10):

| Held-out family | Judge alone: over / mean risk | Router as printed: over / mean risk | Cost-aware router: over / mean risk | Cost-aware, recalibrated on 100 labels of the family |
| --- | --- | --- | --- | --- |
| elements | 0.0% / 0.7% | 0.0% / 0.7% | 0.0% / 1.7% | 0.0% / 2.3% |
| ports | 94.6% / 11.5% | 0.0% / 0.2% | 0.0% / 4.9% | 0.8% / 4.4% |
| unicode | 100% / 69.8% | 0.0% / 1.2% | 100% / 44.0% | 1.0% / 4.0% |
| timezones | 0.0% / 4.4% | 0.0% / 0.5% | 0.0% / 2.9% | 2.0% / 4.5% |

**Search that misses** (`collect.py --nogold`): for a fixed share of items, Mode 2 sees the three nearest neighbours of the answering row and not the row. Without the answering row, the judge rejected 468 of 472 correct element answers and got 183 of 244 time-zone items wrong, 139 of them with uncertainty under 0.1.

| Share of searches missing | Mode 2 accuracy | `joint` over target at α = 0.05 | `joint` kept | `joint_conservative` kept | `joint_split` over / kept |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 99.3% | 0.0% | 99.9% | 99.9% | 0.0% / 99.8% |
| 0.25 | 88.6% | 7.8% | 82.3% | 23.7% | 0.4% / 58.0% |
| 0.5 | 76.7% | 12.8% | 50.7% | 5.4% | 0.4% / 32.2% |

Where the printed search overshot at α = 0.10, the split-certified arm held, at a cost in coverage:

| Condition | `joint` over / kept | `joint_split` over / kept |
| --- | ---: | ---: |
| Half the searches miss; calibrated and scored on ports, unicode, timezones | 12.2% / 57.5% | 0.2% / 35.4% |
| Half the searches miss; calibrated and scored on elements, unicode, timezones | 10.8% / 63.4% | 0.4% / 49.0% |
| A quarter miss; recalibrated on 100 timezone labels | 14.0% / 34.5% | 0.2% / 0.6% |
| A quarter miss; recalibrated on 250 timezone labels | 13.4% / 51.7% | 0.2% / 4.1% |

With few labels from a family where search is unreliable, the honest answer is to keep almost nothing, and that is what the split arm does.

## Limitations

- One judge, `qwen3.5:9b`, grading its own answers. The paper uses four judges and two examinees.
- The questions are factual lookups with exact answers, not the paper's open-domain QA sets.
- Mode 2 search is BM25 over the reference tables, not live web search. The no-gold pass is a crude model of a search that misses.
- `joint_split` and the cost-aware tie-break are mine, not the paper's.
- No official code was available; Algorithm 1, Algorithm 2, Eq. 8 and Eq. 14 are reimplemented from the paper's text. Where the text was ambiguous (renormalising p_true, Eq. 8's prefix condition, the tie-break) the choice is documented in the code.

## Citation

Sher Badshah, Ali Emami and Hassan Sajjad. *Judge, Retrieve, or Abstain: Uncertainty-Guarded LLM Judging with Provable Risk Guarantees.* arXiv:2608.17994, 2026. https://arxiv.org/abs/2608.17994

## License

MIT for the code in this repository. The reference tables are derived from the Unicode Character Database, the IANA service name and port number registry, and the IANA time zone database, under their own terms.
