#!/usr/bin/env python3
"""Re-checks Gate 1's ranking-correlation result at the full meta-dataset
scale, using data that already exists -- no new PURE/FULL_TRAINING compute.

gate1_ranking.py's original result (gradient_norm_variance rho=+0.670,
n=36: 12 tasks x 3 init methods) was the strongest single finding in this
project. But explore_lr_prediction.py just showed the exact same kind of
screening result shrink hard when scaled from n=12 to n=40 (gradient_norm
rho=-0.726 -> -0.343 for LR) -- a small-sample luck effect that a properly
powered check exists to catch. Gate 1's own result has never been checked
this way.

build_meta_dataset.py already ran the *identical* controlled design (§21:
same architecture-generation logic, same FIXED_OPTIMIZER/LEARNING_RATE/
BATCH_SIZE, only init_method varies) across 312 tasks x 3 inits = 936 rows,
already stored in the meta-dataset (both TRAIN and TEST splits pooled --
this is a correlation re-check on already-collected data, not a new
train/test evaluation, so pooling both splits here does not touch the
locked split's own purpose). Recomputing the same Spearman correlations on
that full pool is a direct, free scale-up of exactly the same experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from precog.experiment_db import load_dataframe, record_gate_evaluation
from precog.reporting import export_csv_snapshots, write_report

PROXY_COLUMNS = [
    "zero_cost.synflow", "zero_cost.snip", "zero_cost.grasp", "zero_cost.jacob_cov",
    "zero_cost.effective_rank", "zero_cost.hessian_trace", "zero_cost.jacobian_condition_mean",
    "zero_cost.gradient_norm", "zero_cost.gradient_norm_variance",
    "zero_cost.activation_mean", "zero_cost.activation_variance",
]
NON_CONVERGENCE_PENALTY = 800 * 2
N36_REFERENCE = {  # from results/reports/*_gate1_ranking.md, n=36 (12 tasks)
    "gradient_norm_variance": 0.670, "gradient_norm": 0.607, "jacob_cov": 0.513,
    "effective_rank": 0.483, "snip": 0.467, "jacobian_condition_mean": 0.432,
    "hessian_trace": 0.411, "activation_mean": 0.393, "grasp": -0.380,
    "activation_variance": 0.371, "synflow": 0.270,
}
NAIVE_ZSCORE_N36 = 0.560
RANK_AGGREGATED_N36 = 0.570


def main() -> None:
    full_df = pd.concat([load_dataframe(split="train"), load_dataframe(split="test")], ignore_index=True)
    full_df["steps_to_threshold"] = full_df["steps_to_threshold"].fillna(NON_CONVERGENCE_PENALTY)
    n_tasks = full_df["seed"].nunique()
    print(f"Full pool: {len(full_df)} rows, {n_tasks} tasks (vs n=36 rows / 12 tasks originally)\n")

    true_steps = full_df["steps_to_threshold"].to_numpy()
    results, combined_z, combined_rank = {}, np.zeros(len(full_df)), np.zeros(len(full_df))
    for col in PROXY_COLUMNS:
        name = col.removeprefix("zero_cost.")
        values = full_df[col].to_numpy()
        rho, p = spearmanr(values, true_steps)
        results[name] = (rho, p)
        ref = N36_REFERENCE.get(name)
        shrink = f"  (n=36 was {ref:+.3f}, {'held' if ref and abs(rho) >= 0.8*abs(ref) else 'SHRANK'})" if ref else ""
        print(f"{name:<28} rho={rho:+.3f}  p={p:.3g}{shrink}")
        sign = np.sign(rho or 1)
        combined_z += (values - values.mean()) / (values.std() + 1e-12) * sign
        combined_rank += pd.Series(values * sign).rank().to_numpy()

    rho_z, p_z = spearmanr(combined_z, true_steps)
    rho_rank, p_rank = spearmanr(combined_rank, true_steps)
    print(f"\n{'naive combined (avg z-score)':<28} rho={rho_z:+.3f}  p={p_z:.3g}  (n=36 was {NAIVE_ZSCORE_N36:+.3f})")
    print(f"{'rank-aggregated (AZ-NAS style)':<28} rho={rho_rank:+.3f}  p={p_rank:.3g}  (n=36 was {RANK_AGGREGATED_N36:+.3f})")

    best_name = max(results, key=lambda k: abs(results[k][0]))
    best_rho, best_p = results[best_name]
    gate1_pass = abs(best_rho) >= 0.70
    print(f"\nGate 1 (docs.md §17, |rho|>=0.70) at n={len(full_df)} ({n_tasks} tasks): "
          f"{'PASS' if gate1_pass else 'NOT MET'} (best={best_name}, |rho|={abs(best_rho):.3f})")

    record_gate_evaluation(
        generation="v1-trainability-engine-at-scale", gate_number=1,
        metric_name=f"spearman_rho_{best_name}_vs_steps_full_scale",
        metric_value=float(best_rho), threshold=0.70, n_samples=len(full_df),
        notes=f"full meta-dataset re-check ({n_tasks} tasks) of gate1_ranking.py's original "
              f"n=36 (12-task) result; no new training runs, same controlled design (§21)",
    )

    proxy_table = "\n".join(
        f"| {name} | {rho:+.3f} | {N36_REFERENCE.get(name, float('nan')):+.3f} | "
        f"{'held' if name in N36_REFERENCE and abs(rho) >= 0.8*abs(N36_REFERENCE[name]) else 'shrank'} |"
        for name, (rho, _p) in sorted(results.items(), key=lambda kv: -abs(kv[1][0]))
    )
    report = f"""## Method

Re-checks gate1_ranking.py's ranking-correlation result (n=36: 12 tasks x 3
init methods) at the full meta-dataset scale ({len(full_df)} rows,
{n_tasks} tasks) -- no new PURE/FULL_TRAINING compute, since
build_meta_dataset.py already ran the identical controlled design (§21:
architecture/task-generation logic, FIXED_OPTIMIZER/LEARNING_RATE/
BATCH_SIZE, only init_method varies) at this scale. Motivated directly by
explore_lr_prediction.py's finding that an n=12 screening correlation
(gradient_norm rho=-0.726 for LR) shrank to rho=-0.343 at n=40 -- the same
small-sample risk had never been checked for Gate 1's own headline result.

## Results (each proxy, n={len(full_df)} vs the original n=36)

| proxy | rho (n={len(full_df)}) | rho (n=36) | held or shrank? |
|---|---:|---:|---|
{proxy_table}
| naive combined (avg z-score) | {rho_z:+.3f} | {NAIVE_ZSCORE_N36:+.3f} | {'held' if abs(rho_z) >= 0.8*abs(NAIVE_ZSCORE_N36) else 'shrank'} |
| rank-aggregated (AZ-NAS style) | {rho_rank:+.3f} | {RANK_AGGREGATED_N36:+.3f} | {'held' if abs(rho_rank) >= 0.8*abs(RANK_AGGREGATED_N36) else 'shrank'} |

## Verdict

Gate 1 (docs.md §17, \\|rho\\| >= 0.70) at full scale: **{'PASS' if gate1_pass else 'NOT MET'}**
(best proxy: `{best_name}`, \\|rho\\| = {abs(best_rho):.3f}, p={best_p:.3g}).
"""
    export_csv_snapshots()
    report_path = write_report(
        "gate1_ranking_at_scale", "Gate 1 Re-Check at Full Meta-Dataset Scale", report
    )
    print(f"\nReport written to {report_path.relative_to(report_path.parents[2])}")


if __name__ == "__main__":
    main()
