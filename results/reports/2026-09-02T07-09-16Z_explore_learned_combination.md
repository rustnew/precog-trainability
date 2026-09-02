# Exploring a Learned Zero-Cost Combination (Ridge, LOTO-CV)

_Generated 2026-09-02T07-09-16Z (UTC)_

## Method

Every combination method tried so far (naive z-score averaging, AZ-NAS-style
rank aggregation) uses fixed, equal weights across the 11
zero-cost proxies. This tests whether a *learned* linear combination (Ridge
regression on log1p(steps_to_threshold)) does better -- directly the
question raised in this project's own reflection on regret/OOD results:
"can a learned representation of zero-cost signals generalize better than
individual proxies?"

Evaluated with Leave-One-Task-Out cross-validation, grouped by task (not by
row, since each task contributes 3 correlated rows -- one per init_method
-- so row-level CV would leak). Reuses the meta-dataset's already-stored
rows for the project's 12 standard tasks (36 rows, 3 inits each) -- no new
training runs needed.

## Results

| method | rho (vs steps_to_threshold) |
|---|---:|
| best single proxy (gradient_norm_variance) | +0.670 |
| naive combined (avg z-score) | +0.560 |
| rank-aggregated (AZ-NAS style) | +0.570 |
| **learned linear combination (Ridge, LOTO-CV)** | **+0.504** |
| (reference only, NOT valid generalization) in-sample fit, no held-out task | +0.845 |

### Learned weights (fit on all 12 tasks, for inspection only)

| proxy | Ridge coefficient (standardized) |
|---|---:|
| gradient_norm | +0.858 |
| grasp | +0.608 |
| gradient_norm_variance | +0.487 |
| effective_rank | +0.367 |
| synflow | -0.309 |
| activation_mean | +0.250 |
| hessian_trace | -0.223 |
| snip | -0.176 |
| jacobian_condition_mean | +0.103 |
| jacob_cov | +0.052 |
| activation_variance | -0.041 |

## Verdict

Learned linear combination does NOT beat
the best single proxy under honest (Leave-One-Task-Out) evaluation.
This is the third combination strategy tried (after naive z-score and rank aggregation) and the third to fail against a single well-chosen proxy on this benchmark -- strong, repeated evidence that at this meta-dataset scale (12-40 tasks), combining zero-cost proxies adds noise faster than it adds signal, regardless of how the combination weights are chosen (fixed equal, rank-based, or learned).

The gap between the in-sample fit (rho=+0.845) and the honest
LOTO-CV estimate (rho=+0.504) is itself the point: with only 11
features on 12 tasks, a linear model can fit the training tasks far better
than it generalizes -- exactly the failure mode a proper held-out
evaluation exists to catch, and exactly the risk of judging any future
"learned combination" claim without doing this.
