#!/usr/bin/env python3
"""Every combination method tried so far for the zero-cost proxies (naive
z-score average, AZ-NAS-style rank aggregation) uses *fixed, equal* weights
across proxies -- neither learns which proxies actually deserve more
weight. This tests the natural next step: a learned linear combination
(Ridge regression, steps_to_threshold as target) -- the question the
project's own reflection on regret/OOD raised explicitly: "can a learned
representation of zero-cost signals generalize better than individual
proxies?"

Evaluated with Leave-One-Task-Out cross-validation (not in-sample fit,
which would trivially "win" by overfitting 36 rows with 11 features) --
grouped by task (not by row) since each task contributes 3 correlated rows
(one per init_method), so a naive row-level CV would leak a task's other
two rows into its own held-out fold.

Reuses the meta-dataset's already-stored rows for the project's 12 standard
tasks (gate1_ranking.py's own controlled-experiment tasks) -- no new
PURE/FULL_TRAINING compute needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from precog.experiment_db import load_dataframe, record_gate_evaluation
from precog.reporting import export_csv_snapshots, write_report

PROXY_COLUMNS = [
    "zero_cost.synflow", "zero_cost.snip", "zero_cost.grasp", "zero_cost.jacob_cov",
    "zero_cost.effective_rank", "zero_cost.hessian_trace", "zero_cost.jacobian_condition_mean",
    "zero_cost.gradient_norm", "zero_cost.gradient_norm_variance",
    "zero_cost.activation_mean", "zero_cost.activation_variance",
]

# Reference numbers from results/reports/*_gate1_ranking.md (same 12 tasks,
# 36 rows, controlled per §21) for direct comparison.
BEST_INDIVIDUAL_PROXY = ("gradient_norm_variance", 0.670)
NAIVE_ZSCORE_COMBINED = 0.560
RANK_AGGREGATED_COMBINED = 0.570


def main() -> None:
    df = load_dataframe(split="train")
    df = df[df["seed"].between(1, 12)].reset_index(drop=True)
    print(f"Loaded {len(df)} rows, {df['seed'].nunique()} tasks (the project's 12 standard tasks)\n")

    x = df[PROXY_COLUMNS].to_numpy()
    y_steps = df["steps_to_threshold"].to_numpy()
    y_log = np.log1p(y_steps)
    groups = df["seed"].to_numpy()

    logo = LeaveOneGroupOut()
    oof_predictions = np.zeros(len(df))
    for train_idx, test_idx in logo.split(x, y_log, groups):
        scaler = StandardScaler().fit(x[train_idx])
        model = Ridge(alpha=1.0)
        model.fit(scaler.transform(x[train_idx]), y_log[train_idx])
        oof_predictions[test_idx] = model.predict(scaler.transform(x[test_idx]))

    rho_learned, p_learned = spearmanr(oof_predictions, y_steps)
    print(f"Learned linear combination (Ridge, Leave-One-Task-Out CV): "
          f"rho={rho_learned:+.3f}  p={p_learned:.3g}")
    print(f"vs naive z-score combined:      rho={NAIVE_ZSCORE_COMBINED:+.3f}")
    print(f"vs AZ-NAS rank-aggregated:      rho={RANK_AGGREGATED_COMBINED:+.3f}")
    print(f"vs best single proxy ({BEST_INDIVIDUAL_PROXY[0]}): rho={BEST_INDIVIDUAL_PROXY[1]:+.3f}")

    # Also fit on ALL data (in-sample, no held-out task) to show the gap
    # between what looks achievable in-sample vs what actually generalizes
    # -- the overfitting risk this LOTO design exists to catch.
    scaler_full = StandardScaler().fit(x)
    model_full = Ridge(alpha=1.0)
    model_full.fit(scaler_full.transform(x), y_log)
    in_sample_predictions = model_full.predict(scaler_full.transform(x))
    rho_in_sample, _ = spearmanr(in_sample_predictions, y_steps)
    print(f"\n(For reference only, NOT a valid generalization estimate) "
          f"in-sample fit (no held-out task): rho={rho_in_sample:+.3f}")

    beats_best_single = abs(rho_learned) > abs(BEST_INDIVIDUAL_PROXY[1])
    coef_table = "\n".join(
        f"| {col.removeprefix('zero_cost.')} | {coef:+.3f} |"
        for col, coef in sorted(zip(PROXY_COLUMNS, model_full.coef_), key=lambda kv: -abs(kv[1]))
    )

    record_gate_evaluation(
        generation="v1-learned-combination", gate_number=1,
        metric_name="spearman_rho_ridge_loto_cv_combined_vs_steps",
        metric_value=float(rho_learned), threshold=0.70, n_samples=len(df),
        notes=f"Leave-One-Task-Out CV Ridge regression on {len(PROXY_COLUMNS)} zero-cost proxies, "
              f"p={p_learned:.3g}, in_sample_rho={rho_in_sample:+.3f} (overfitting reference only), "
              f"beats_best_single_proxy={beats_best_single}",
    )

    report = f"""## Method

Every combination method tried so far (naive z-score averaging, AZ-NAS-style
rank aggregation) uses fixed, equal weights across the {len(PROXY_COLUMNS)}
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
| best single proxy (gradient_norm_variance) | {BEST_INDIVIDUAL_PROXY[1]:+.3f} |
| naive combined (avg z-score) | {NAIVE_ZSCORE_COMBINED:+.3f} |
| rank-aggregated (AZ-NAS style) | {RANK_AGGREGATED_COMBINED:+.3f} |
| **learned linear combination (Ridge, LOTO-CV)** | **{rho_learned:+.3f}** |
| (reference only, NOT valid generalization) in-sample fit, no held-out task | {rho_in_sample:+.3f} |

### Learned weights (fit on all 12 tasks, for inspection only)

| proxy | Ridge coefficient (standardized) |
|---|---:|
{coef_table}

## Verdict

Learned linear combination {'beats' if beats_best_single else 'does NOT beat'}
the best single proxy under honest (Leave-One-Task-Out) evaluation.
{"This is the third combination strategy tried (after naive z-score and rank aggregation) and the third to fail against a single well-chosen proxy on this benchmark -- strong, repeated evidence that at this meta-dataset scale (12-40 tasks), combining zero-cost proxies adds noise faster than it adds signal, regardless of how the combination weights are chosen (fixed equal, rank-based, or learned)." if not beats_best_single else "Unlike every fixed-weight combination tried so far, a properly cross-validated learned combination does add value over the single best proxy -- worth carrying into the Meta-Predictor's zero-cost feature aggregation."}

The gap between the in-sample fit (rho={rho_in_sample:+.3f}) and the honest
LOTO-CV estimate (rho={rho_learned:+.3f}) is itself the point: with only 11
features on 12 tasks, a linear model can fit the training tasks far better
than it generalizes -- exactly the failure mode a proper held-out
evaluation exists to catch, and exactly the risk of judging any future
"learned combination" claim without doing this.
"""
    export_csv_snapshots()
    report_path = write_report(
        "explore_learned_combination", "Exploring a Learned Zero-Cost Combination (Ridge, LOTO-CV)", report
    )
    print(f"\nReport written to {report_path.relative_to(report_path.parents[2])}")


if __name__ == "__main__":
    main()
