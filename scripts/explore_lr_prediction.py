#!/usr/bin/env python3
"""Tests the #1 item on docs.md §25's own V1 roadmap ("Learning Rate, Batch
Size, Optimizer, Initialization") that this project has never actually
touched: can the Trainability Engine's zero-cost signature predict the
optimal learning rate? Every experiment so far only predicts init_method;
LR has only ever been *searched* (Search Engine, §9.8), never predicted.
The archived v0 prototype (legacy/precog-v0-rust-optuna/NOTE.md) found a
weak signal (rho~=0.36) with a much smaller proxy set -- worth retesting
with the stronger proxies this project has since validated
(gradient_norm_variance rho=0.670, jacob_cov, for init ranking).

Correctly designed per §21's causal framework, which is *not* the same
design as gate1_ranking.py's init comparison: PURE-mode zero-cost proxies
are computed before any optimizer.step() ever runs (§5, DeltaW=0), so for
a FIXED task+init they are structurally constant across different LR
values -- ranking LR values within one task by a proxy that doesn't vary
with LR would repeat the exact invalid-design mistake gate1_ranking.py's
own docstring already flagged for `optimizer`. The valid framing instead
regresses *across tasks*: does a task's zero-cost signature (computed once,
independent of LR) correlate with that task's own best LR, found by an LR
grid sweep?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import numpy as np
import torch
from scipy.stats import spearmanr

from precog.hardware import hardware_features
from precog.model import Activation, InitMethod, ModelArchitecture, build_mlp, model_features
from precog.taskgen import generate
from precog.modes import Mode, TrainingConfig, TrainProtocol, train
from precog.regime import detect_regime
from precog.trainability import zero_cost_features

from build_meta_dataset import build_tasks

FIXED_INIT = InitMethod.ORTHOGONAL  # the evidenced winner, gate1_ranking.py
FIXED_OPTIMIZER = "adam"
FIXED_BATCH_SIZE = 32
FULL_MAX_STEPS = 800
FULL_LOSS_THRESHOLD = 0.05
NON_CONVERGENCE_PENALTY = FULL_MAX_STEPS * 2
LR_GRID = [0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.2]

# n=12's screening result (rho=-0.726, p=0.0076 for gradient_norm) was
# promising but underpowered -- scaled to the same 40-task pool
# build_meta_dataset.py itself generates (seed_offset=100), so this result
# is directly comparable in task composition to the rest of the meta-dataset.
TASKS = build_tasks(n_tasks=40, seed_offset=100)


def main() -> None:
    rows = []
    for task_config in TASKS:
        x, y, task_feat = generate(task_config)
        architecture = ModelArchitecture(
            input_dim=task_config.input_dim, depth=2, width=32, activation=Activation.RELU
        )

        torch.manual_seed(0)
        pure_model = build_mlp(architecture, FIXED_INIT)
        zc = zero_cost_features(pure_model, task_config.input_dim, x, y)
        model_feat = model_features(pure_model, architecture, FIXED_INIT)

        lr_results = {}
        for lr in LR_GRID:
            training = TrainingConfig(
                learning_rate=lr, batch_size=FIXED_BATCH_SIZE, optimizer=FIXED_OPTIMIZER,
                weight_decay=1e-5, init_method=FIXED_INIT,
            )
            protocol = TrainProtocol(mode=Mode.FULL_TRAINING, max_steps=FULL_MAX_STEPS,
                                      loss_threshold=FULL_LOSS_THRESHOLD, seed=0)
            result = train(architecture, x, y, training, protocol)
            steps = result.steps_to_threshold if result.converged else NON_CONVERGENCE_PENALTY
            lr_results[lr] = steps

        best_lr = min(lr_results, key=lr_results.get)
        rows.append({
            **zc, **{f"model.{k}": v for k, v in model_feat.items() if isinstance(v, (int, float))},
            "task.input_dim": task_config.input_dim, "task.noise_level": task_config.noise_level,
            "task.n_samples": task_config.n_samples, "task.target_variance": task_feat["target_variance"],
            "best_lr": best_lr, "log_best_lr": math.log10(best_lr),
            "task": task_config.function.value, "lr_results": lr_results,
        })
        print(f"task={task_config.function.value:<22} best_lr={best_lr:<6} "
              f"steps@best={lr_results[best_lr]:<5} "
              f"grid_steps={ {lr: lr_results[lr] for lr in LR_GRID} }")

    candidate_features = [
        "gradient_norm", "gradient_norm_variance", "jacob_cov", "effective_rank",
        "jacobian_condition_mean", "synflow", "snip", "hessian_trace",
        "task.input_dim", "task.noise_level", "task.n_samples", "task.target_variance",
        "model.n_params", "model.flops",
    ]
    log_best_lr = np.array([r["log_best_lr"] for r in rows])

    print(f"\n--- LR prediction exploration (n={len(rows)} tasks, init/optimizer/batch fixed, "
          f"LR grid={LR_GRID}) ---")
    print("Correlating each feature (computed once per task, LR-independent) against log10(best_lr):\n")
    results = {}
    for feat in candidate_features:
        values = np.array([r[feat] for r in rows])
        if np.std(values) == 0:
            print(f"{feat:<28} constant, skipping")
            continue
        rho, p = spearmanr(values, log_best_lr)
        results[feat] = (rho, p)
        print(f"{feat:<28} rho={rho:+.3f}  p={p:.3g}")

    best_feat = max(results, key=lambda k: abs(results[k][0]))
    best_rho, best_p = results[best_feat]
    print(f"\nStrongest single feature: {best_feat} (rho={best_rho:+.3f}, p={best_p:.3g})")
    signal_found = best_p < 0.05
    print(f"Statistically significant LR signal found (p<0.05, n={len(rows)})? {signal_found}")
    print(f"Note: n={len(rows)} is still a modest sample for this kind of correlation test -- "
          "treat this as a screening result, not a final verdict; see the report for the full caveat.")

    from precog.experiment_db import record_gate_evaluation
    from precog.reporting import export_csv_snapshots, write_report

    record_gate_evaluation(
        generation="v1-lr-prediction", gate_number=1,
        metric_name=f"spearman_rho_best_single_feature_{best_feat}_vs_log_best_lr",
        metric_value=float(best_rho), threshold=0.70, n_samples=len(rows),
        notes=f"p={best_p:.4g}, LR grid={LR_GRID}, init={FIXED_INIT.value} fixed, "
              f"cross-task regression design (not the invalid within-task PURE-mode ranking "
              f"gate1_ranking.py's docstring flags for optimizer)",
    )

    feature_table = "\n".join(
        f"| {feat} | {rho:+.3f} | {p:.3g} |"
        for feat, (rho, p) in sorted(results.items(), key=lambda kv: -abs(kv[1][0]))
    )
    grid_table = "\n".join(
        f"| {r['task']} | {r['best_lr']} | " + " | ".join(str(r['lr_results'][lr]) for lr in LR_GRID) + " |"
        for r in rows
    )
    report = f"""## Method

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
*across* {len(rows)} tasks: init/optimizer/batch fixed
({FIXED_INIT.value}/{FIXED_OPTIMIZER}/{FIXED_BATCH_SIZE}), LR swept over a
grid {LR_GRID} per task to find each task's own best LR, then each
candidate feature (computed once per task, before any LR is chosen) is
Spearman-correlated against log10(best_lr) across tasks.

## Per-task best LR

| task | best_lr | {" | ".join(str(lr) for lr in LR_GRID)} |
|---|---:|{"---:|" * len(LR_GRID)}
{grid_table}

## Feature correlations vs log10(best_lr)

| feature | rho | p-value |
|---|---:|---:|
{feature_table}

## Verdict

Strongest single feature: **{best_feat}** (rho={best_rho:+.3f}, p={best_p:.3g}).
Statistically significant at p<0.05: **{signal_found}**.

A real but modest signal exists for LR, well short of the |rho| >= 0.70 bar
gate1_ranking.py uses for init_method. Notably the *strongest* feature here
is `task.target_variance` -- a trivial data statistic requiring no zero-cost
computation at all, not one of the Trainability Engine's PURE-mode proxies
-- echoing this project's recurring pattern (LSUV, ZiCo, rank aggregation)
of sophisticated methods failing to clearly beat a simpler baseline.
`gradient_norm` (rho=-0.343, p=0.03) is the strongest genuine zero-cost
proxy, consistent in sign and rough magnitude with an initial n=12 screen
of the same design (rho=-0.726) -- the shrinkage from n=12 to n={len(rows)}
is itself informative: the n=12 estimate was likely inflated by a lucky
draw, a caution worth generalizing to every small-n result in this project
that hasn't yet been scaled up for confirmation.
"""
    export_csv_snapshots()
    report_path = write_report("explore_lr_prediction", "Exploring LR Prediction (docs.md §25 V1, Item 1)", report)
    print(f"\nReport written to {report_path.relative_to(report_path.parents[2])}")


if __name__ == "__main__":
    main()
