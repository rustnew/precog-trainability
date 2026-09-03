# Exploring Active Sample Selection (Pillar 6: Sample Efficiency)

_Generated 2026-09-01T14-37-08Z (UTC)_

## Method

Tests active sample selection (hard-example mining: periodically rescore
every training sample by current per-sample loss, restrict subsequent
batches to the hardest 50%, refreshed every 8
steps) against `precog.modes.train()`'s existing uniform-random batching --
source.md pillar 6 (Active Learning / Sample Efficiency), never tested in
this project before. Controlled per §21: architecture, task, learning rate
(0.02), batch size (32), optimizer
(adam) and init (orthogonal, the evidenced winner from
gate1_ranking.py) are all fixed and identical between arms across the same
12 synthetic tasks used throughout this project. Since batch size is
identical in both arms, N_epsilon (samples to threshold, docs.md §16) is
just steps_to_threshold * batch_size, so a reduction in steps is a
reduction in N_epsilon by construction.

## Results

| task | random steps | active steps | random N_epsilon | active N_epsilon | winner |
|---|---:|---:|---:|---:|---|
| linear | 20 | 30 | 640 | 960 | random |
| nonlinear_interaction | 115 | 221 | 3680 | 7072 | random |
| nonlinear_product | 264 | 303 | 8448 | 9696 | random |
| linear | 42 | 1600 | 1344 | 51200 | random |
| nonlinear_interaction | 81 | 165 | 2592 | 5280 | random |
| nonlinear_product | 250 | 1600 | 8000 | 51200 | random |
| linear | 23 | 24 | 736 | 768 | random |
| nonlinear_interaction | 266 | 494 | 8512 | 15808 | random |
| nonlinear_product | 191 | 478 | 6112 | 15296 | random |
| linear | 41 | 46 | 1312 | 1472 | random |
| nonlinear_interaction | 100 | 184 | 3200 | 5888 | random |
| nonlinear_product | 85 | 288 | 2720 | 9216 | random |

**Active-sampling win rate: 0%** (0/12 tasks)
**Mean N_epsilon:** random=3941, active=14488
(-267.6% change, positive = active reduces samples needed)

## Verdict

Active sampling does NOT clearly reduce
N_epsilon on this benchmark (NEGATIVE RESULT
against the >50% win-rate + net-reduction bar). Consistent with the project's overall finding on these small, low-dimensional synthetic regression tasks: most sophistication tested so far (LSUV init, gradient_alignment proxy) has failed to beat simpler baselines -- the tasks may simply be too easy/low-dimensional for hard-example mining to matter, since even random batches already cover the input space densely at these sample sizes ({[t.n_samples for t in TASKS]}).
