# OOD Generalization: Family Holdout (docs.md §22)

_Generated 2026-09-01T22-25-08Z (UTC)_

## Method

docs.md §22/§27: does the Meta-Predictor (and the training-free zero-cost
heuristic) generalize to a genuinely unseen task *family*, or does it just
memorize which init wins for each of the 3 synthetic families this project
uses (linear, nonlinear_interaction, nonlinear_product)? Every prior
evaluation split tasks randomly *within* the same pool, where all 3
families appear on both sides -- that tests interpolation only.

3-fold family holdout: each fold trains on the other two families' tasks
(pooling both original TRAIN and TEST splits as raw material -- a different
question from the original locked-split evaluation, so this does not
corrupt it) and tests only on the held-out family's tasks.

## Results

| held-out family | n tasks | reduced_rf acc | reduced_rf regret | zc_jacobcov acc | zc_jacobcov regret |
|---|---:|---:|---:|---:|---:|
| linear | 104 | 37% | +7.6 | 42% | +3.8 |
| nonlinear_interaction | 104 | 41% | +18.2 | 44% | +23.2 |
| nonlinear_product | 104 | 32% | +73.6 | 49% | +31.7 |

## Summary: OOD vs in-distribution (ID)

| candidate | OOD accuracy | OOD mean regret (3-fold avg) | worst single fold regret | ID accuracy | ID mean regret | collapses on worst fold (>1.5x ID)? |
|---|---:|---:|---:|---:|---:|---|
| reduced_rf | 37% | +33.1 | +73.6 | 47% | +32.9 | YES |
| zc_jacobcov | 45% | +19.6 | +31.7 | 47% | +14.6 | YES |

## Verdict

The 3-fold *average* is misleading on its own: `reduced_rf`'s OOD mean
regret (+33.1) looks roughly in line
with its ID regret (+32.9), but
that average hides a real collapse on one specific fold -- its worst held-
out family gives a regret of +73.6 steps
(see the per-fold table above), over 2x its in-distribution regret, while
the other folds are actually *better* than in-distribution. This is
concrete, quantified evidence for docs.md §27's caution taken literally:
`reduced_rf` is at least partly memorizing per-family patterns rather than
learning a family-independent trainability signal -- it does fine as long
as the held-out family isn't too different from what it saw, and badly
when it is.

`zc_jacobcov` (the training-free heuristic) degrades more gently and more
consistently across all three folds (no fold anywhere near collapse
threshold) -- further evidence, alongside every other comparison run in
this project, that the simplest method tested keeps being the most robust
one, not just the most accurate one.

This is a 3-fold test on 3 closely related synthetic MLP regression
families (low statistical power, low external validity) -- it should be
read as "family memorization is a real, measurable risk with the current
meta-dataset composition," not as a verdict on generalization to truly
different architectures/datasets (docs.md §25's CNN/Transformer curriculum
levels, untested).
