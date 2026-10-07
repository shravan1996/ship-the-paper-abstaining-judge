# Pre-registration

Written and committed before the full collection run and before any threshold was calibrated.
Before writing it I ran a 9-item pilot to check that the prompts parse and that Ollama returns
log-probabilities at the verdict token. Those rows are not part of the dataset and no statistic
was computed from them.

## The paper

Sher Badshah, Ali Emami and Hassan Sajjad. *Judge, Retrieve, or Abstain: Uncertainty-Guarded LLM
Judging with Provable Risk Guarantees.* arXiv:2608.17994, first version 18 August 2026. Accepted
at COLM 2026.

The claim under test: thresholds calibrated on held-out labels keep the false discovery rate of
accepted verdicts at or below a target α with probability at least 1 − δ (Eq. 4, Eq. 15), and
the two-threshold router (Algorithm 1 offline, Algorithm 2 online) accepts substantially more
verdicts than the judge alone (Table 2).

## Question

Calibrated on 1,000 labelled verdicts, does the paper's two-threshold router keep the share of
wrong verdicts among the verdicts it accepts at or below α in at least 1 − δ of calibration
draws, while accepting at least 10 points more verdicts than the judge alone?

## Setup

- **Population.** 1,972 factual questions in four families: elements (472), ports (500),
  unicode (500), timezones (500). Reference answers come from tables (the periodic table,
  `/etc/services`, Python's `unicodedata` 16.0, the system tzdata 2026c). Labels are exact
  matches after normalisation, so a label is never a model's opinion.
- **Examinee and judge.** `qwen3.5:9b` served by Ollama 0.34.4, temperature 0, thinking off,
  seed 7. The examinee's answer is the candidate. The same model judges it.
- **Uncertainty.** Binary predictive entropy of the renormalised True and False probabilities at
  the verdict token, as in Section 4 and Appendix B.2.
- **Mode 2 evidence.** BM25 top 3 over 15,772 reference documents, formatted as index, title,
  snippet and source.
- **Truth for the guarantee.** The population's empirical distribution. Each calibration set is
  1,000 items drawn i.i.d. with replacement from the population, and the risk of the thresholds
  it selects is computed exactly on the whole population. The i.i.d. assumption therefore holds
  by construction, and any shortfall belongs to the procedure, not to the data.
- **Draws.** 1,000 per condition for the primary analysis, 500 for the secondary ones.
  α ∈ {0.05, 0.10, 0.15, 0.20, 0.25}, δ = 0.05. **Primary α = 0.10.**

## Arms

| Arm | What it is |
| --- | --- |
| `mode1_only` | Eq. 8 on Mode 1 uncertainty: the judge alone. **Baseline.** |
| `mode2_only` | Eq. 8 on Mode 2 uncertainty: always retrieve. |
| `joint` | Algorithm 1, pointwise Clopper-Pearson at δ, as published. **Pre-registered arm.** |
| `joint_conservative` | Algorithm 1 with every check at δ / (\|T1\| \|T2\|), Appendix D.4. |
| `joint_split` | Exploratory, my addition: choose candidate pairs on half the calibration set, certify them in fixed order on the other half. Not part of the decision. |

## Metrics

- **Violation rate:** share of draws whose selected thresholds have population risk above α.
  This is what Eq. 4 bounds by δ.
- **Mean population risk:** what the paper reports in Figure 1.
- **Coverage:** share of population items with an accepted verdict.
- Retrieval share, abstention share, and seconds per item from the recorded durations.

## Decision rule (primary α = 0.10, `joint` against `mode1_only`)

- **Continue** if the violation rate is at most 6.5% (δ plus two Monte Carlo standard errors at
  1,000 draws) **and** mean coverage is at least 10 points above `mode1_only`.
- **Stop** if the violation rate is 10% or more, **or** the coverage gain is under 5 points.
- Anything else is inconclusive.

## Secondary analyses (reported, not part of the decision)

1. **New family.** Calibrate on three families (1,000 i.i.d. draws from their union) and
   measure population risk on the fourth.
2. **Recalibration.** Calibrate on 100 or 250 labels drawn from the new family alone.
3. **Calibration size.** n ∈ {250, 500, 1,000}.
4. **Paper-style splits.** 100 random 50/50 splits, mean and standard deviation of empirical
   test FDR, to compare with the way Figure 1 reports it.

## Predictions

1. `mode1_only` holds: violation rate at most 5% at every α.
2. `joint` violates more often than δ, between 8% and 25% at α = 0.10, because Algorithm 1 keeps
   the best of many pairs that were each tested at δ, while Lemma 1 covers one fixed pair.
3. `joint` coverage beats `mode1_only` by at least 20 points at α = 0.10, because retrieval here
   nearly always returns the right row.
4. `joint_conservative` holds, but gives up at least half of `joint`'s coverage gain.
5. New family: for at least one held-out family, `joint`'s population risk exceeds α in more
   than half the draws.
6. Most Mode 1 failures are True verdicts on the examinee's own wrong answers.
