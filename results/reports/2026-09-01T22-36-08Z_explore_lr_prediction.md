# Exploring LR Prediction (docs.md §25 V1, Item 1)

_Generated 2026-09-01T22-36-08Z (UTC)_

## Method

docs.md §25's V1 roadmap lists Learning Rate first, but every experiment in
this project so far has only predicted `init_method` -- LR has only ever
been searched (Search Engine, §9.8), never predicted from a task's zero-cost
signature. The archived v0 prototype found a weak signal (rho~=0.36) with a
much smaller, weaker proxy set; this retests the question with the stronger
proxies this project has since validated for init ranking.

Design note (why this differs from gate1_ranking.py): PURE-mode zero-cost
proxies are computed before any optimizer.step() (§5), so for one fixed
task+init they cannot vary with LR at all -- ranking LR values *within* a
task by a LR-independent proxy would repeat gate1_ranking.py's own
previously-fixed mistake for `optimizer`. The valid design instead regresses
*across* 40 tasks: init/optimizer/batch fixed
(orthogonal/adam/32), LR swept over a
grid [0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.2] per task to find each task's own best LR, then each
candidate feature (computed once per task, before any LR is chosen) is
Spearman-correlated against log10(best_lr) across tasks.

## Per-task best LR

| task | best_lr | 0.001 | 0.003 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 0.05 | 81 | 30 | 20 | 21 | 18 | 20 | 61 |
| nonlinear_interaction | 0.05 | 1600 | 355 | 226 | 225 | 186 | 630 | 1600 |
| nonlinear_product | 0.02 | 1600 | 482 | 400 | 266 | 1600 | 1600 | 1600 |
| linear | 0.01 | 242 | 106 | 66 | 87 | 84 | 126 | 149 |
| nonlinear_interaction | 0.02 | 1600 | 507 | 254 | 242 | 335 | 1600 | 1600 |
| nonlinear_product | 0.02 | 693 | 280 | 183 | 155 | 210 | 1600 | 1600 |
| linear | 0.05 | 124 | 42 | 32 | 22 | 18 | 24 | 63 |
| nonlinear_interaction | 0.05 | 628 | 405 | 245 | 190 | 153 | 241 | 1600 |
| nonlinear_product | 0.05 | 646 | 295 | 170 | 133 | 81 | 494 | 1600 |
| linear | 0.02 | 83 | 25 | 19 | 12 | 17 | 27 | 82 |
| nonlinear_interaction | 0.02 | 668 | 216 | 248 | 163 | 219 | 1600 | 1600 |
| nonlinear_product | 0.02 | 1600 | 1600 | 451 | 381 | 1600 | 1600 | 1600 |
| linear | 0.01 | 220 | 75 | 28 | 39 | 52 | 50 | 90 |
| nonlinear_interaction | 0.02 | 1600 | 389 | 237 | 146 | 190 | 457 | 261 |
| nonlinear_product | 0.01 | 1600 | 404 | 213 | 258 | 461 | 222 | 1600 |
| linear | 0.02 | 78 | 42 | 18 | 12 | 22 | 15 | 46 |
| nonlinear_interaction | 0.02 | 601 | 293 | 196 | 147 | 199 | 376 | 693 |
| nonlinear_product | 0.02 | 650 | 413 | 212 | 203 | 234 | 357 | 1600 |
| linear | 0.05 | 82 | 31 | 21 | 21 | 18 | 20 | 37 |
| nonlinear_interaction | 0.02 | 476 | 266 | 133 | 109 | 194 | 297 | 1600 |
| nonlinear_product | 0.02 | 1600 | 481 | 209 | 165 | 323 | 1600 | 1600 |
| linear | 0.01 | 66 | 25 | 8 | 10 | 16 | 12 | 40 |
| nonlinear_interaction | 0.02 | 434 | 210 | 149 | 89 | 113 | 465 | 1600 |
| nonlinear_product | 0.01 | 1600 | 446 | 155 | 203 | 493 | 1600 | 1600 |
| linear | 0.05 | 112 | 51 | 26 | 28 | 15 | 29 | 71 |
| nonlinear_interaction | 0.05 | 276 | 159 | 65 | 79 | 58 | 88 | 216 |
| nonlinear_product | 0.01 | 1600 | 676 | 285 | 285 | 1600 | 1600 | 1600 |
| linear | 0.01 | 98 | 43 | 41 | 41 | 56 | 43 | 43 |
| nonlinear_interaction | 0.1 | 443 | 282 | 123 | 123 | 191 | 116 | 357 |
| nonlinear_product | 0.01 | 1600 | 1600 | 458 | 458 | 1600 | 1600 | 1600 |
| linear | 0.02 | 189 | 61 | 40 | 28 | 30 | 39 | 83 |
| nonlinear_interaction | 0.05 | 337 | 132 | 89 | 66 | 40 | 124 | 781 |
| nonlinear_product | 0.02 | 747 | 296 | 129 | 98 | 139 | 356 | 1600 |
| linear | 0.05 | 119 | 60 | 27 | 17 | 15 | 30 | 89 |
| nonlinear_interaction | 0.02 | 534 | 174 | 97 | 80 | 129 | 270 | 416 |
| nonlinear_product | 0.01 | 1600 | 748 | 364 | 364 | 1600 | 1600 | 1600 |
| linear | 0.05 | 164 | 69 | 38 | 32 | 30 | 35 | 47 |
| nonlinear_interaction | 0.02 | 467 | 262 | 121 | 113 | 113 | 187 | 198 |
| nonlinear_product | 0.02 | 771 | 345 | 189 | 138 | 189 | 252 | 1600 |
| linear | 0.02 | 202 | 93 | 93 | 37 | 202 | 213 | 110 |

## Feature correlations vs log10(best_lr)

| feature | rho | p-value |
|---|---:|---:|
| task.target_variance | -0.417 | 0.00748 |
| gradient_norm | -0.343 | 0.03 |
| snip | -0.324 | 0.0411 |
| task.n_samples | -0.168 | 0.301 |
| gradient_norm_variance | -0.142 | 0.384 |
| task.noise_level | -0.137 | 0.4 |
| hessian_trace | +0.100 | 0.541 |
| jacobian_condition_mean | +0.098 | 0.547 |
| synflow | +0.074 | 0.648 |
| task.input_dim | +0.074 | 0.648 |
| model.n_params | +0.074 | 0.648 |
| model.flops | +0.074 | 0.648 |
| effective_rank | +0.048 | 0.77 |
| jacob_cov | +0.004 | 0.978 |

## Verdict

Strongest single feature: **task.target_variance** (rho=-0.417, p=0.00748).
Statistically significant at p<0.05: **True**.

A real but modest signal exists for LR, well short of the |rho| >= 0.70 bar
gate1_ranking.py uses for init_method. Notably the *strongest* feature here
is `task.target_variance` -- a trivial data statistic requiring no zero-cost
computation at all, not one of the Trainability Engine's PURE-mode proxies
-- echoing this project's recurring pattern (LSUV, ZiCo, rank aggregation)
of sophisticated methods failing to clearly beat a simpler baseline.
`gradient_norm` (rho=-0.343, p=0.03) is the strongest genuine zero-cost
proxy, consistent in sign and rough magnitude with an initial n=12 screen
of the same design (rho=-0.726) -- the shrinkage from n=12 to n=40
is itself informative: the n=12 estimate was likely inflated by a lucky
draw, a caution worth generalizing to every small-n result in this project
that hasn't yet been scaled up for confirmation.
