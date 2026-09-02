# PRECOG — Can Pre-Training Signals Predict Trainability?

[![Reproduce](https://github.com/rustnew/precog-trainability/actions/workflows/reproduce.yml/badge.svg)](https://github.com/rustnew/precog-trainability/actions/workflows/reproduce.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)

A research prototype testing whether "zero-cost" signals computed on an
**untrained** network (before a single optimizer step) can predict which
hyperparameters will actually train well. Full spec in [docs.md](docs.md).

This is a curated subset of a larger project. It exists to show three
things honestly, in order: **a real positive finding, a headline number
that turned out to be wrong and how we caught it, and a mechanistic bug we
found but could not fix.** If you work in NAS / zero-cost proxies /
meta-learning, the last two sections are where we'd like your help.

## TL;DR

- Best method found: **`jacob_cov`** (NASWOT, Mellor et al. 2021), used
  directly as a decision rule with **zero training, zero learned model** —
  beats 4 RandomForest variants, a Gaussian Process, KNN, and 3 different
  ways of combining proxies. 47% top-1 accuracy picking the best of 3
  init methods (baseline: 38%), and a much lower decision-regret than
  every alternative (+14.6 mean steps vs +32.9 for the best learned model).
- Our own headline ranking-correlation number (ρ=0.670) was a **small-sample
  overestimate**. Rechecked at 26x the sample size: ρ=0.395-0.540. We kept
  both results and the script that found the discrepancy — see below.
- `jacob_cov` (and 2 other proxies) can **provably never** recommend one of
  our 3 candidate init methods, for a reason that's mathematical, not a
  data problem. We could not fix it. Details below.

## 1. What actually works

`precog/trainability.py` implements 11 zero-cost proxies (SynFlow, SNIP,
GraSP, Jacob-Cov/NASWOT, effective rank, Hessian trace, Jacobian
conditioning, gradient statistics, ZiCo). `scripts/compare_meta_predictors.py`
evaluates every method — RandomForest (full/reduced features, log-target),
Gaussian Process, k-NN, raw proxy heuristics, learned/rank/naive proxy
combinations — on one **locked** test split (60 tasks never touched during
development), reporting both top-1 accuracy and **regret**
(`steps(predicted) - steps(true_best)`, since a wrong call that costs 5
extra steps and one that costs 700 are not the same mistake).

Full comparison table: [results/reports/...compare_meta_predictors.md](results/reports/2026-09-02T07-46-49Z_compare_meta_predictors.md)

The winner, every time we've re-run this, is the simplest thing on the
list: rank the 3 candidates by their raw `jacob_cov` score, no model at
all. Every attempt to do better with more sophistication failed:

| Combination method | Ranking correlation (ρ) |
|---|---:|
| naive z-score average | 0.365 |
| AZ-NAS-style rank aggregation (CVPR 2024) | 0.410 |
| learned linear combination (Ridge, proper Leave-One-Task-Out CV) | 0.504 |
| **best single proxy alone** | **0.540** |

The Ridge result is the one we think is most worth a second pair of eyes:
in-sample (no held-out task) it reaches ρ=0.845 — a textbook illustration
of overfitting with 11 features on a dozen tasks — which is exactly why we
insist on Leave-One-**Task**-Out grouping, not row-level CV (each task
contributes 3 correlated rows, one per candidate). Full writeup:
[results/reports/...explore_learned_combination.md](results/reports/2026-09-02T07-09-16Z_explore_learned_combination.md)

## 2. The number we got wrong, and how we caught it

Our first controlled experiment (`scripts/gate1_ranking.py`, n=36: 12 tasks
x 3 init methods) found `gradient_norm_variance` correlating with real
convergence speed at **ρ=+0.670** ([report](results/reports/2026-09-01T22-12-09Z_gate1_ranking.md)).
We used this number in design decisions and reported it as the project's
strongest result.

It didn't hold. `scripts/gate1_ranking_at_scale.py` reruns the identical
correlation on the full meta-dataset (n=936: 312 tasks, same controlled
design, zero new training runs — the data already existed) and finds
**ρ=0.395** for that same proxy. The actual best individual proxy at this
scale is `gradient_norm` (ρ=0.540, down modestly from 0.607). Every proxy
shrinks by some amount; two combination methods shrink too.
[Full correction](results/reports/2026-09-02T07-10-47Z_gate1_ranking_at_scale.md).

We saw the same pattern independently for learning-rate prediction: a
screen at n=12 found `gradient_norm` correlating with optimal LR at
ρ=-0.726 (p=0.0076); scaled to n=40, it dropped to ρ=-0.343 (still
significant, far weaker). [Report](results/reports/2026-09-01T22-36-08Z_explore_lr_prediction.md).

We're flagging this loudly because we don't think we're unusual — a lot of
n=12-36 zero-cost-proxy correlation numbers get published and cited without
ever being rechecked at 10-30x the sample size. Ours didn't survive that
check unscathed and we'd rather say so than not.

## 3. A bug we found but can't fix

While digging into *why* our best method (`jacob_cov`) never once
recommends "He init" across 60 test tasks — even the 10 where He genuinely
is fastest — we found the cause is not statistical, it's structural:

`jacob_cov` (NASWOT) scores a network by the binary activation-sign
pattern it produces on a batch (`sign(activation) > 0`). Xavier and He
initialization, for a network with zero-initialized biases, draw from
*the same underlying Gaussian random values* (same shape, same global RNG
state under the hood) and only differ in the **positive scalar** they
scale those values by. Scaling every layer's weights by a positive
constant does not change the sign of any activation. So:

```
jacob_cov(He-init network)  ==  jacob_cov(Xavier-init network)
```

exactly, for the same random seed. We checked: max difference across all
312 tasks in our meta-dataset is `0.0`. Not close to zero — exactly zero.
Two other proxies we use (`effective_rank`, entropy of *normalized*
singular values; `jacobian_condition_mean`, a singular-value *ratio*) turn
out to have the same property for the same reason — any proxy built from a
scale-invariant statistic cannot see this distinction.
[Full audit of all 11 proxies](results/reports/2026-09-02T08-04-49Z_explore_scale_invariance_blindspot.md).

We tried two fixes:
1. Break jacob_cov ties with `gradient_norm` (which *does* differ by
   init). Failed — `gradient_norm(He)` turned out to be higher than
   `gradient_norm(Xavier)` on **all 312/312 tasks** (a fixed ~2.3x scale
   ratio from He's larger init std), so the tiebreak picks Xavier
   deterministically too, just for a different reason.
2. Z-score `gradient_norm` against each init family's own population
   statistics before comparing (removing that fixed offset). This made
   things *worse* (33% accuracy vs 47%), suggesting there's close to no
   real task-dependent signal left in `gradient_norm` once you remove the
   scale confound.

We don't have a third idea from the zero-cost literature we haven't
already tried. If your first thought is "why not proxy X" — there's a
real chance we tried it; check the [full report](results/reports/2026-09-02T08-04-49Z_explore_scale_invariance_blindspot.md)
first. If it's a proxy family we haven't touched (something sensitive to
task-dependent absolute scale, not just a fixed per-init offset), that's
exactly the kind of comment we're hoping for.

## 4. Generalization: does it know anything, or does it just remember?

We held out one entire synthetic task family from training and tested only
on it (3 folds). The learned RandomForest's regret on its worst held-out
fold is **2.2x its in-distribution regret** (+73.6 vs +32.9 mean steps) —
concrete evidence it's partly memorizing per-family patterns, not learning
a family-independent notion of trainability. The training-free `jacob_cov`
heuristic degrades far more gently across every fold.
[Report](results/reports/2026-09-01T22-25-08Z_explore_ood_generalization.md).

## Reproduce it

```bash
pip install -r requirements.txt
python scripts/gate1_ranking.py              # the original (overestimated) result
python scripts/gate1_ranking_at_scale.py     # the correction, on data already in data/meta_dataset.db
python scripts/explore_scale_invariance_blindspot.py   # the Xavier/He proof
python scripts/compare_meta_predictors.py    # full method comparison + regret
```

`data/meta_dataset.db` ships with the repo (2.5MB, 312 tasks / 936
controlled experiments) so the scale-corrected results reproduce without
rerunning any training.

## What this is not

Not a finished system. No gate in the full spec's success criteria is met
yet (see [docs.md §17, §26](docs.md)). This is a snapshot of an ongoing,
openly-documented research effort — the full project (40+ dated reports,
every negative result kept) lives at
[github.com/NEURAX-canvas/PRECOG](https://github.com/NEURAX-canvas/PRECOG).

## Repository layout

```text
docs.md / stack.md / source.md   full spec, tech-stack rationale, bibliography
precog/                          the engine: encoders, Trainability Engine,
                                  Meta-Predictor, Search Engine (see docs.md §9)
scripts/                         every experiment referenced above, runnable as-is
results/reports/                 the dated, evidence-first writeup for each finding
data/meta_dataset.db             312 tasks x 936 controlled experiments (SQLite)
.github/workflows/reproduce.yml  CI: reruns every script above on each push
```

## Citing this work

See [CITATION.cff](CITATION.cff). This is an active research snapshot, not
a peer-reviewed publication — cite accordingly.

## License

[MIT](LICENSE).
