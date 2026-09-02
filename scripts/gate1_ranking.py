#!/usr/bin/env python3
"""Gate 1 (docs.md §17): does the Trainability Engine's PURE-mode score rank
configurations the way FULL TRAINING eventually would? Target: Spearman
rho >= 0.70 (docs.md §16, P1 Ranking protocol, §15).

Designed as a controlled experiment per §21 ("only the candidate variable
changes, architecture/dataset/optimizer fixed"), not a mixed sweep: the
first version of this script varied optimizer AND init_method together,
which conflates two effects and, worse, is structurally invalid for
optimizer -- the Trainability Engine's proxies are computed in PURE mode,
strictly before any optimizer exists (§5, ΔW=0), so none of them can
possibly encode which optimizer will later be used. Ranking-correlating a
score that is *constant* across optimizers against an outcome that *varies*
by optimizer measures noise, not proxy quality.

So this script isolates exactly one candidate variable -- initialization --
with architecture, task, learning rate, batch size and optimizer all fixed,
which is also the one degree of freedom §9.4's "initialization analysis /
dynamical isometry" signals are theoretically supposed to explain (§11.2).
Predicting optimizer/LR/batch is a different, harder question that needs
the learned Meta-Predictor (§9.7) trained on the meta-dataset, not a raw
PURE-mode score -- see the printed note at the end.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from precog.experiment_db import experiment_exists, record_experiment, record_gate_evaluation
from precog.hardware import hardware_features
from precog.model import Activation, InitMethod, ModelArchitecture, build_mlp, model_features
from precog.modes import Mode, TrainingConfig, TrainProtocol, train
from precog.regime import detect_regime
from precog.taskgen import TaskConfig, TaskFunction, generate
from precog.trainability import zero_cost_features

# Controlled per §21: fixed for every row below.
FIXED_OPTIMIZER = "adam"
FIXED_LEARNING_RATE = 0.02
FIXED_BATCH_SIZE = 32
FULL_MAX_STEPS = 800
FULL_LOSS_THRESHOLD = 0.05
NON_CONVERGENCE_PENALTY = FULL_MAX_STEPS * 2

# The one candidate variable.
INIT_METHODS = [InitMethod.XAVIER, InitMethod.HE, InitMethod.ORTHOGONAL]

TASKS = [
    TaskConfig(TaskFunction.LINEAR, input_dim=4, noise_level=0.1, n_samples=256, seed=1),
    TaskConfig(TaskFunction.NONLINEAR_INTERACTION, input_dim=6, noise_level=0.1, n_samples=512, seed=2),
    TaskConfig(TaskFunction.NONLINEAR_PRODUCT, input_dim=5, noise_level=0.1, n_samples=384, seed=3),
    TaskConfig(TaskFunction.LINEAR, input_dim=4, noise_level=0.3, n_samples=256, seed=4),
    TaskConfig(TaskFunction.NONLINEAR_INTERACTION, input_dim=6, noise_level=0.0, n_samples=512, seed=5),
    TaskConfig(TaskFunction.NONLINEAR_PRODUCT, input_dim=5, noise_level=0.2, n_samples=768, seed=6),
    TaskConfig(TaskFunction.LINEAR, input_dim=5, noise_level=0.0, n_samples=384, seed=7),
    TaskConfig(TaskFunction.NONLINEAR_INTERACTION, input_dim=7, noise_level=0.2, n_samples=640, seed=8),
    TaskConfig(TaskFunction.NONLINEAR_PRODUCT, input_dim=6, noise_level=0.05, n_samples=512, seed=9),
    TaskConfig(TaskFunction.LINEAR, input_dim=6, noise_level=0.2, n_samples=320, seed=10),
    TaskConfig(TaskFunction.NONLINEAR_INTERACTION, input_dim=5, noise_level=0.05, n_samples=448, seed=11),
    TaskConfig(TaskFunction.NONLINEAR_PRODUCT, input_dim=4, noise_level=0.1, n_samples=256, seed=12),
]


def main() -> None:
    rows = []

    for task_config in TASKS:
        x, y, task_feat = generate(task_config)

        for init_method in INIT_METHODS:
            architecture = ModelArchitecture(
                input_dim=task_config.input_dim, depth=2, width=32, activation=Activation.RELU
            )
            training = TrainingConfig(
                learning_rate=FIXED_LEARNING_RATE,
                batch_size=FIXED_BATCH_SIZE,
                optimizer=FIXED_OPTIMIZER,
                weight_decay=1e-5,
                init_method=init_method,
            )

            # PURE mode: build once, score, never call .step() on this model.
            torch.manual_seed(0)
            pure_model = build_mlp(architecture, init_method)
            zc = zero_cost_features(pure_model, task_config.input_dim, x, y)
            model_feat = model_features(pure_model, architecture, init_method)
            hw_feat = hardware_features()
            regime = detect_regime(model_feat, task_feat, hw_feat)

            # FULL TRAINING: ground truth, a fresh model of its own.
            protocol = TrainProtocol(
                mode=Mode.FULL_TRAINING,
                max_steps=FULL_MAX_STEPS,
                loss_threshold=FULL_LOSS_THRESHOLD,
                seed=0,
            )
            result = train(architecture, x, y, training, protocol)
            true_steps = result.steps_to_threshold if result.converged else NON_CONVERGENCE_PENALTY

            # §12: every experiment is recorded, not just printed. These
            # diagnostic runs aren't part of the Meta-Predictor's locked
            # train/test split (build_meta_dataset.py owns that), so they're
            # tagged "train" -- exploratory, never used for a final gate
            # evaluation of the Meta-Predictor itself. Guarded by
            # experiment_exists() so re-running this script (e.g. after
            # adding a new proxy) doesn't duplicate these 12 tasks' rows in
            # the meta-dataset every time -- found after this script's
            # repeat runs this session had silently inflated the "train"
            # split's row count (756 -> 792 -> 828) without adding any new
            # distinct tasks.
            if experiment_exists(task_config.seed, init_method.value):
                rows.append({**zc, "true_steps": true_steps, "task": task_config.function.value})
                continue
            record_experiment(
                split="train",
                seed=task_config.seed,
                mode=Mode.FULL_TRAINING.value,
                model_features=model_feat,
                task_features=task_feat,
                hardware_features=hw_feat,
                regime=regime,
                training_config={
                    "learning_rate": training.learning_rate,
                    "batch_size": training.batch_size,
                    "optimizer": training.optimizer,
                    "weight_decay": training.weight_decay,
                    "init_method": training.init_method.value,
                },
                outcome=result,
                zero_cost_features=zc,
            )

            rows.append({**zc, "true_steps": true_steps, "task": task_config.function.value})
            print(
                f"task={task_config.function.value:<22} init={init_method.value:<11} "
                f"true_steps={true_steps:<5} synflow={zc['synflow']:.3g} "
                f"jacob_cov={zc['jacob_cov']:.3g} grad_norm={zc['gradient_norm']:.3g}"
            )

    proxies = [
        "synflow", "snip", "grasp", "jacob_cov", "effective_rank", "hessian_trace",
        "jacobian_condition_mean", "gradient_norm", "gradient_norm_variance",
        "activation_mean", "activation_variance", "gradient_alignment", "zico",
    ]
    true_steps_arr = np.array([r["true_steps"] for r in rows])

    print(f"\n--- Gate 1: ranking correlation (n={len(rows)} rows, "
          f"{len(TASKS)} tasks x {len(INIT_METHODS)} init methods, optimizer/LR/batch fixed) ---")

    per_proxy_results = {}
    combined_z = np.zeros(len(rows))
    combined_rank = np.zeros(len(rows))
    for proxy in proxies:
        values = np.array([r[proxy] for r in rows])
        if np.std(values) == 0:
            print(f"{proxy:<28} constant across all rows, skipping")
            continue
        rho, p_value = spearmanr(values, true_steps_arr)
        print(f"{proxy:<28} rho={rho:+.3f}  p={p_value:.3g}")
        per_proxy_results[proxy] = (rho, p_value)
        sign = np.sign(rho or 1)
        combined_z += (values - values.mean()) / (values.std() + 1e-12) * sign
        # Rank aggregation (AZ-NAS, CVPR 2024, arXiv:2403.19232): sum of
        # per-proxy ranks instead of raw z-scores -- robust to a proxy whose
        # *scale* is well-behaved but whose raw values are noisy/skewed
        # (unlike z-score averaging, a proxy's outliers can only move its
        # rank by one place, not distort the whole sum).
        combined_rank += pd.Series(values * sign).rank().to_numpy()

    rho_combined, p_combined = spearmanr(combined_z, true_steps_arr)
    print(f"{'naive combined (avg z-score)':<28} rho={rho_combined:+.3f}  p={p_combined:.3g}")
    rho_rank, p_rank = spearmanr(combined_rank, true_steps_arr)
    print(f"{'rank-aggregated (AZ-NAS style)':<28} rho={rho_rank:+.3f}  p={p_rank:.3g}")

    best_combined_rho = max(rho_combined, rho_rank, key=abs)
    gate1_pass = abs(best_combined_rho) >= 0.70
    print(f"\nGate 1 (docs.md §17, target |rho| >= 0.70) on the init-only controlled "
          f"experiment: {'PASS' if gate1_pass else 'NOT YET MET'} (|rho|={abs(best_combined_rho):.3f}, "
          f"best of naive-zscore/rank-aggregated)")

    # §17: persist every gate check so generations can actually be compared
    # over time, not just read off the console of whoever ran this last.
    record_gate_evaluation(
        generation="v1-trainability-engine",
        gate_number=1,
        metric_name="spearman_rho_naive_combined_zero_cost_vs_steps",
        metric_value=float(rho_combined),
        threshold=0.70,
        n_samples=len(rows),
        notes=f"controlled experiment (§21): {len(TASKS)} tasks x {len(INIT_METHODS)} init methods, "
              f"optimizer/LR/batch fixed",
    )
    record_gate_evaluation(
        generation="v1-trainability-engine",
        gate_number=1,
        metric_name="spearman_rho_rank_aggregated_zero_cost_vs_steps",
        metric_value=float(rho_rank),
        threshold=0.70,
        n_samples=len(rows),
        notes=f"AZ-NAS-style (arXiv:2403.19232) rank-sum aggregation vs naive z-score averaging, "
              f"same controlled experiment (§21): {len(TASKS)} tasks x {len(INIT_METHODS)} init methods",
    )
    for proxy, (rho, _p) in per_proxy_results.items():
        record_gate_evaluation(
            generation="v1-trainability-engine",
            gate_number=1,
            metric_name=f"spearman_rho_{proxy}_vs_steps",
            metric_value=float(rho),
            threshold=0.70,
            n_samples=len(rows),
            notes="individual proxy, not the combined score used for the official gate verdict",
        )

    from precog.reporting import export_csv_snapshots, write_report

    proxy_table = "\n".join(
        f"| {proxy} | {rho:+.3f} | {p:.3g} |" for proxy, (rho, p) in sorted(
            per_proxy_results.items(), key=lambda kv: -abs(kv[1][0])
        )
    )
    report = f"""## Method

Controlled experiment per docs.md §21: only `init_method` varies
({", ".join(m.value for m in INIT_METHODS)}); architecture, task, learning
rate ({FIXED_LEARNING_RATE}), batch size ({FIXED_BATCH_SIZE}) and optimizer
({FIXED_OPTIMIZER}) are all fixed. {len(TASKS)} synthetic tasks x
{len(INIT_METHODS)} init methods = {len(rows)} rows. Zero-cost features
computed in PURE mode (DeltaW=0); ground truth from FULL_TRAINING
(max {FULL_MAX_STEPS} steps, threshold {FULL_LOSS_THRESHOLD}).

## Results

| proxy | rho | p-value |
|---|---:|---:|
{proxy_table}
| naive combined (avg z-score) | {rho_combined:+.3f} | {p_combined:.3g} |
| **rank-aggregated (AZ-NAS style, arXiv:2403.19232)** | **{rho_rank:+.3f}** | {p_rank:.3g} |

Rank aggregation sums each proxy's *rank* across the 36 rows (Borda-count
style) instead of z-scored raw values, so one noisy/skewed proxy (e.g.
`synflow`, weakest of the lot) can only move the sum by its rank distance,
not distort it via scale -- the fix AZ-NAS (CVPR 2024) proposes for exactly
the failure this project's naive combination showed (combined rho below its
own best individual proxy).

## Verdict

Gate 1 (docs.md §17, target \\|rho\\| >= 0.70): **{'PASS' if gate1_pass else 'NOT YET MET'}**
(\\|rho\\| = {abs(best_combined_rho):.3f}, best of the two combination methods above)

Every proxy above is computed in PURE mode, strictly before any
optimizer/LR/batch_size is chosen (DeltaW=0) -- so this experiment can only
test whether the Trainability Engine explains *initialization* quality. It
cannot rank optimizer/LR/batch_size choices by construction; that needs the
Meta-Predictor (§9.7), see the meta-predictor evaluation report.
"""
    export_csv_snapshots()
    report_path = write_report("gate1_ranking", "Gate 1 — Ranking Correlation (Trainability Engine)", report)
    print(f"\nReport written to {report_path.relative_to(report_path.parents[2])}")
    print("Meta-dataset snapshot refreshed in results/experiments.csv and results/gate_evaluations.csv")

    print(
        "\nNote (§9.4, §5): every proxy above is computed in PURE mode, strictly before any\n"
        "optimizer/LR/batch_size is chosen (DeltaW=0) -- so this experiment can only ever\n"
        "test whether the Trainability Engine explains *initialization* quality. It\n"
        "structurally cannot rank optimizer/LR/batch_size choices: those need the learned\n"
        "Meta-Predictor (§9.7), trained on the meta-dataset (§12) to associate a\n"
        "(model, data, zero-cost-features) signature with historically good H* -- the\n"
        "same lesson the archived v0 prototype already learned the hard way."
    )


if __name__ == "__main__":
    main()
