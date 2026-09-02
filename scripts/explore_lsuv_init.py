#!/usr/bin/env python3
"""Explores LSUV (data-aware init, source.md pillar 4) as a 4th candidate
alongside Xavier/He/Orthogonal, in the same controlled design as
gate1_ranking.py (§21: architecture, task, LR, batch size and optimizer all
fixed, only init_method varies) -- does calibrating the init empirically
against the task's actual data beat the purely analytic formulas?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from precog.experiment_db import record_gate_evaluation
from precog.model import Activation, InitMethod, ModelArchitecture, build_mlp
from precog.modes import Mode, TrainingConfig, TrainProtocol, train
from precog.reporting import export_csv_snapshots, write_report
from precog.taskgen import TaskConfig, TaskFunction, generate

FIXED_OPTIMIZER = "adam"
FIXED_LEARNING_RATE = 0.02
FIXED_BATCH_SIZE = 32
FULL_MAX_STEPS = 800
FULL_LOSS_THRESHOLD = 0.05
NON_CONVERGENCE_PENALTY = FULL_MAX_STEPS * 2

INIT_METHODS = [InitMethod.XAVIER, InitMethod.HE, InitMethod.ORTHOGONAL, InitMethod.LSUV]

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
        x, y, _ = generate(task_config)
        architecture = ModelArchitecture(
            input_dim=task_config.input_dim, depth=2, width=32, activation=Activation.RELU
        )
        per_init = {}
        for init_method in INIT_METHODS:
            training = TrainingConfig(
                learning_rate=FIXED_LEARNING_RATE, batch_size=FIXED_BATCH_SIZE, optimizer=FIXED_OPTIMIZER,
                weight_decay=1e-5, init_method=init_method,
            )
            protocol = TrainProtocol(mode=Mode.FULL_TRAINING, max_steps=FULL_MAX_STEPS,
                                      loss_threshold=FULL_LOSS_THRESHOLD, seed=0)

            # LSUV needs build_mlp to see the data for calibration; the other
            # three ignore data_sample entirely (still PURE-mode compliant,
            # §5 -- these are forward passes for calibration, never a
            # gradient-descent update).
            if init_method == InitMethod.LSUV:
                torch.manual_seed(0)
                calibration_model = build_mlp(architecture, InitMethod.LSUV, data_sample=x[: min(64, len(x))])
                # modes.train() builds its own fresh model internally; to
                # actually use the LSUV-calibrated weights we train this
                # model directly rather than going through modes.train().
                result = _train_prebuilt(calibration_model, x, y, training, protocol)
            else:
                result = train(architecture, x, y, training, protocol)

            steps = result.steps_to_threshold if result.converged else NON_CONVERGENCE_PENALTY
            per_init[init_method.value] = steps

        best_init = min(per_init, key=per_init.get)
        rows.append({**per_init, "task": task_config.function.value, "best": best_init})
        print(f"task={task_config.function.value:<22} " +
              "  ".join(f"{k}={v:.0f}" for k, v in per_init.items()) + f"  best={best_init}")

    print(f"\n--- LSUV exploration ({len(rows)} tasks, controlled per §21) ---")
    wins = {m.value: sum(1 for r in rows if r["best"] == m.value) for m in INIT_METHODS}
    means = {m.value: float(np.mean([r[m.value] for r in rows])) for m in INIT_METHODS}
    for m in INIT_METHODS:
        print(f"{m.value:<12} wins={wins[m.value]}/{len(rows)}  mean_steps={means[m.value]:.0f}")

    lsuv_beats_all_analytic = means["lsuv"] < min(means["xavier"], means["he"], means["orthogonal"])
    record_gate_evaluation(
        generation="v1-lsuv-init", gate_number=1, metric_name="lsuv_wins_rate",
        metric_value=wins["lsuv"] / len(rows), threshold=1 / len(INIT_METHODS), n_samples=len(rows),
        notes=f"means={means}, lsuv_beats_all_analytic_on_average={lsuv_beats_all_analytic}",
    )

    detail_rows = "\n".join(
        f"| {r['task']} | " + " | ".join(f"{r[m.value]:.0f}" for m in INIT_METHODS) + f" | {r['best']} |"
        for r in rows
    )
    report = f"""## Method

LSUV (data-aware init, Mishkin & Matas 2015, source.md pillar 4) added as a
4th candidate alongside Xavier/He/Orthogonal, controlled per §21 (task,
architecture, LR={FIXED_LEARNING_RATE}, batch={FIXED_BATCH_SIZE},
optimizer={FIXED_OPTIMIZER} all fixed, only init_method varies), across
{len(rows)} synthetic tasks (the same 12 as gate1_ranking.py).

## Results

| task | xavier | he | orthogonal | lsuv | best |
|---|---:|---:|---:|---:|---|
{detail_rows}

| init | wins | mean steps |
|---|---:|---:|
{chr(10).join(f"| {m.value} | {wins[m.value]}/{len(rows)} | {means[m.value]:.0f} |" for m in INIT_METHODS)}

## Verdict

LSUV beats all three analytic inits on average steps: **{lsuv_beats_all_analytic}**.
Random chance for any one init to win a given task: {100/len(INIT_METHODS):.0f}%.
"""
    export_csv_snapshots()
    report_path = write_report("explore_lsuv_init", "Exploring LSUV (Data-Aware Init)", report)
    print(f"\nReport written to {report_path.relative_to(report_path.parents[2])}")


def _train_prebuilt(model, x, y, training, protocol):
    """Trains an already-built (already-initialized) model, mirroring
    precog.modes.train()'s loop exactly but skipping the internal
    build_mlp() call so LSUV's calibrated weights are the ones trained."""
    import time

    from precog.modes import TrainResult

    torch.manual_seed(protocol.seed)
    generator = torch.Generator().manual_seed(protocol.seed)
    initial_weights = torch.cat([p.detach().flatten() for p in model.parameters()])

    if training.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=training.learning_rate)
    elif training.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate, weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)

    loss_fn = torch.nn.functional.mse_loss
    n_samples = x.shape[0]
    batch_size = min(training.batch_size, n_samples)

    steps_to_threshold = None
    initial_loss = float("nan")
    final_loss = float("nan")
    diverged = False
    start = time.perf_counter()

    for step in range(protocol.max_steps):
        if step % max(1, n_samples // batch_size) == 0:
            order = torch.randperm(n_samples, generator=generator)
        batch_idx = order[(step * batch_size) % n_samples: (step * batch_size) % n_samples + batch_size]
        if batch_idx.numel() == 0:
            batch_idx = order[:batch_size]

        pred = model(x[batch_idx])
        loss = loss_fn(pred, y[batch_idx])
        loss_value = loss.item()
        if step == 0:
            initial_loss = loss_value
        if not torch.isfinite(loss).item():
            diverged = True
            final_loss = loss_value
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = loss_value
        if steps_to_threshold is None and loss_value < protocol.loss_threshold:
            steps_to_threshold = step + 1
            break

    wall_clock_s = time.perf_counter() - start
    final_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
    return TrainResult(
        mode=protocol.mode, initial_loss=initial_loss, final_loss=final_loss,
        steps_to_threshold=steps_to_threshold, converged=steps_to_threshold is not None,
        diverged=diverged, wall_clock_s=wall_clock_s,
        delta_w_norm=float((final_weights - initial_weights).norm().item()),
    )


if __name__ == "__main__":
    main()
