# PROBE-Mode Tie-Break for the Xavier/He Blind Spot

_Generated 2026-09-03T12-59-38Z (UTC)_

## Method

Three candidates evaluated once each on the identical locked TEST split
(180 rows, 60 tasks) used throughout this
project:

- `zc_jacobcov` -- the raw heuristic, already known to never recommend "he"
  (0/10 on tasks where "he" is truly best).
- `zc_jacobcov_tiebreak` -- the first static-secondary-proxy fix attempt
  (gradient_norm), from scripts/compare_meta_predictors.py.
- `zc_jacobcov_probetiebreak` -- this run's new candidate
  (precog.meta_predictor.ProbeTieBreakPredictor): on the exact tie
  jacob_cov cannot break, spends a real, bounded PROBE-mode budget
  (50 steps per tied candidate, docs.md §5) and picks whichever
  ends with the lower loss, instead of another PURE-mode secondary proxy.

Per the Zero-Training Contract (docs.md §5: "must always be possible to
answer how much PROBE adds over PURE alone, for what additional cost"),
the table below reports that cost -- mean extra training steps spent per
decision -- next to accuracy, he-recall (of the 10
test tasks where "he" is genuinely the fastest choice) and regret.

## Results

| candidate | accuracy | he-recall | mean regret (steps) | mean probe cost (steps) |
|---|---:|---:|---:|---:|
| zc_jacobcov | 47% (28/60) | 0% (0/10) | +14.6 | 0.0 |
| zc_jacobcov_tiebreak | 47% (28/60) | 0% (0/10) | +14.6 | 0.0 |
| zc_jacobcov_probetiebreak | 43% (26/60) | 30% (3/10) | +45.4 | 68.3 |

PROBE overhead: 68 steps/decision on average,
against a mean 182-step FULL TRAINING run on this
split (37.5% of it). Since jacob_cov ties on every
single test task (the blind spot is structural, not occasional), this is
also the candidate's *total* added cost -- there is no untied case to
amortize it against.

| seed | true best init | predicted (probetiebreak) | regret (steps) | probe cost (steps) |
|---|---|---|---:|---:|
| 100 | xavier | orthogonal | 10 | 0 |
| 107 | orthogonal | orthogonal | 0 | 0 |
| 120 | orthogonal | he | 149 | 100 |
| 131 | xavier | xavier | 0 | 100 |
| 132 | xavier | xavier | 0 | 100 |
| 137 | orthogonal | xavier | 7 | 100 |
| 141 | orthogonal | he | 128 | 100 |
| 146 | orthogonal | orthogonal | 0 | 0 |
| 147 | xavier | he | 973 | 100 |
| 148 | he | orthogonal | 1 | 0 |
| 150 | he | he | 0 | 100 |
| 151 | orthogonal | orthogonal | 0 | 0 |
| 155 | orthogonal | xavier | 54 | 100 |
| 171 | orthogonal | orthogonal | 0 | 0 |
| 172 | he | he | 0 | 100 |
| 175 | orthogonal | xavier | 3 | 100 |
| 197 | xavier | xavier | 0 | 100 |
| 204 | xavier | he | 281 | 100 |
| 211 | orthogonal | xavier | 1 | 100 |
| 213 | xavier | xavier | 0 | 100 |
| 222 | orthogonal | orthogonal | 0 | 0 |
| 224 | xavier | xavier | 0 | 100 |
| 228 | xavier | xavier | 0 | 100 |
| 232 | he | xavier | 8 | 100 |
| 233 | orthogonal | xavier | 71 | 100 |
| 244 | xavier | he | 9 | 100 |
| 249 | he | orthogonal | 55 | 0 |
| 254 | orthogonal | orthogonal | 0 | 0 |
| 255 | orthogonal | orthogonal | 0 | 0 |
| 258 | xavier | orthogonal | 7 | 0 |
| 261 | xavier | xavier | 0 | 100 |
| 263 | orthogonal | xavier | 66 | 100 |
| 266 | orthogonal | he | 49 | 100 |
| 269 | orthogonal | xavier | 9 | 100 |
| 270 | he | xavier | 8 | 100 |
| 281 | he | xavier | 9 | 100 |
| 283 | orthogonal | orthogonal | 0 | 0 |
| 297 | orthogonal | xavier | 46 | 100 |
| 304 | orthogonal | orthogonal | 0 | 0 |
| 307 | orthogonal | xavier | 8 | 100 |
| 315 | xavier | xavier | 0 | 100 |
| 322 | orthogonal | xavier | 1 | 100 |
| 326 | xavier | xavier | 0 | 100 |
| 329 | xavier | xavier | 0 | 100 |
| 341 | xavier | xavier | 0 | 100 |
| 344 | orthogonal | xavier | 131 | 100 |
| 348 | he | orthogonal | 18 | 0 |
| 350 | orthogonal | xavier | 218 | 100 |
| 352 | xavier | xavier | 0 | 100 |
| 358 | he | orthogonal | 8 | 0 |
| 360 | he | he | 0 | 100 |
| 361 | orthogonal | he | 9 | 100 |
| 366 | xavier | orthogonal | 7 | 0 |
| 372 | xavier | orthogonal | 3 | 0 |
| 378 | orthogonal | orthogonal | 0 | 0 |
| 380 | xavier | xavier | 0 | 100 |
| 382 | xavier | he | 5 | 100 |
| 386 | xavier | orthogonal | 0 | 0 |
| 390 | xavier | he | 214 | 100 |
| 398 | orthogonal | he | 156 | 100 |

## Verdict

He-recall rises from
0% (0/10) to
30% (3/10) with the PROBE
tie-break -- unlike the two static secondary-proxy attempts, spending real
(bounded) training budget on the exact tie can see the Xavier/He difference
at all, because it is the only one of the three that isn't a
positive-rescaling-invariant statistic by construction.

But regret -- this project's primary metric, precisely because it catches
what accuracy alone can't (scripts/compare_meta_predictors.py) -- gets
**worse**, not better: +14.6 steps (raw) to
+45.4 steps (PROBE tie-break), on top of
68 extra steps/decision spent
(37.5% of a mean FULL TRAINING run) to get there.
50 steps is enough to sometimes make "he" look locally better
than Xavier, but not enough to foresee the cases named in this project's
own README §1 where an init that looks fine early on never actually
converges within budget: the single worst case here is seed 147
(true best is xavier at 627 steps, the
probe picks "he", which needs 1600 steps to
reach threshold -- regret +973). Net: **NOT a net improvement**
-- it trades a narrow, cosmetic fix (jacob_cov's 'never recommends he' symptom) for a worse net decision rule, the same shape of finding as this project's other ideas that looked reasonable but underperformed a simpler baseline (LSUV init, active sampling) -- see source.md pillar 4 and §5 above.
