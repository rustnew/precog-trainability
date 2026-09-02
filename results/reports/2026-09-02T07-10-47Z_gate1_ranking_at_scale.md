# Gate 1 Re-Check at Full Meta-Dataset Scale

_Generated 2026-09-02T07-10-47Z (UTC)_

## Method

Re-checks gate1_ranking.py's ranking-correlation result (n=36: 12 tasks x 3
init methods) at the full meta-dataset scale (936 rows,
312 tasks) -- no new PURE/FULL_TRAINING compute, since
build_meta_dataset.py already ran the identical controlled design (§21:
architecture/task-generation logic, FIXED_OPTIMIZER/LEARNING_RATE/
BATCH_SIZE, only init_method varies) at this scale. Motivated directly by
explore_lr_prediction.py's finding that an n=12 screening correlation
(gradient_norm rho=-0.726 for LR) shrank to rho=-0.343 at n=40 -- the same
small-sample risk had never been checked for Gate 1's own headline result.

## Results (each proxy, n=936 vs the original n=36)

| proxy | rho (n=936) | rho (n=36) | held or shrank? |
|---|---:|---:|---|
| gradient_norm | +0.540 | +0.607 | held |
| grasp | -0.489 | -0.380 | held |
| snip | +0.438 | +0.467 | held |
| gradient_norm_variance | +0.395 | +0.670 | shrank |
| jacobian_condition_mean | +0.333 | +0.432 | shrank |
| jacob_cov | +0.319 | +0.513 | shrank |
| effective_rank | +0.309 | +0.483 | shrank |
| synflow | +0.239 | +0.270 | held |
| activation_mean | +0.231 | +0.393 | shrank |
| activation_variance | +0.215 | +0.371 | shrank |
| hessian_trace | +0.173 | +0.411 | shrank |
| naive combined (avg z-score) | +0.365 | +0.560 | shrank |
| rank-aggregated (AZ-NAS style) | +0.410 | +0.570 | shrank |

## Verdict

Gate 1 (docs.md §17, \|rho\| >= 0.70) at full scale: **NOT MET**
(best proxy: `gradient_norm`, \|rho\| = 0.540, p=6.25e-72).
