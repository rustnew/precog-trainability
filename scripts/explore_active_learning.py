#!/usr/bin/env python3
"""Explores active sample selection (source.md pillar 6, docs.md §16 "data
efficiency") -- a completely different axis from everything tested so far.
Every other script in this project asks "how many *steps* (T_epsilon) does
it take to reach the loss threshold?". This one asks "how many *samples*
(N_epsilon) does the model need to see?" -- docs.md §16/§17 name both as
first-class efficiency metrics, but N_epsilon had never actually been
measured until now.

Controlled per §21: architecture, task, learning rate, batch size, optimizer
and init_method are all fixed and identical between arms; the only thing
that varies is the sampling policy used to build each training batch:
  - random:  precog.modes.train()'s existing torch.randperm batching.
  - active:  precog.active_learning.train_active()'s hard-example mining --
             periodically rescore every sample by current loss, restrict
             subsequent batches to the hardest top_fraction of them.
Since batch_size is identical in both arms, samples_seen_to_threshold is
just steps_to_threshold * batch_size, so a win on steps is a win on
N_epsilon by construction here -- the interesting question is only whether
active selection ever reduces steps_to_threshold at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from precog.active_learning import train_active
from precog.experiment_db import record_gate_evaluation
from precog.model import Activation, InitMethod, ModelArchitecture
from precog.modes import Mode, TrainingConfig, TrainProtocol, train
from precog.reporting import export_csv_snapshots, write_report
from precog.taskgen import TaskConfig, TaskFunction, generate

FIXED_OPTIMIZER = "adam"
FIXED_LEARNING_RATE = 0.02
FIXED_BATCH_SIZE = 32
FIXED_INIT = InitMethod.ORTHOGONAL  # the evidenced winner (gate1_ranking.py)
FULL_MAX_STEPS = 800
FULL_LOSS_THRESHOLD = 0.05
NON_CONVERGENCE_PENALTY = FULL_MAX_STEPS * 2
TOP_FRACTION = 0.5
REFRESH_EVERY = 8

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
        training = TrainingConfig(
            learning_rate=FIXED_LEARNING_RATE, batch_size=FIXED_BATCH_SIZE, optimizer=FIXED_OPTIMIZER,
            weight_decay=1e-5, init_method=FIXED_INIT,
        )
        protocol = TrainProtocol(mode=Mode.FULL_TRAINING, max_steps=FULL_MAX_STEPS,
                                  loss_threshold=FULL_LOSS_THRESHOLD, seed=0)

        random_result = train(architecture, x, y, training, protocol)
        random_steps = random_result.steps_to_threshold if random_result.converged else NON_CONVERGENCE_PENALTY
        random_samples = random_steps * FIXED_BATCH_SIZE

        active_result = train_active(
            architecture, x, y, training, protocol, top_fraction=TOP_FRACTION, refresh_every=REFRESH_EVERY,
        )
        active_steps = active_result.steps_to_threshold if active_result.converged else NON_CONVERGENCE_PENALTY
        active_samples = active_steps * FIXED_BATCH_SIZE

        rows.append({
            "task": task_config.function.value,
            "random_steps": random_steps, "active_steps": active_steps,
            "random_samples": random_samples, "active_samples": active_samples,
            "active_wins": active_samples < random_samples,
        })
        print(f"task={task_config.function.value:<22} random_steps={random_steps:<5} "
              f"active_steps={active_steps:<5} active_wins={active_samples < random_samples}")

    n = len(rows)
    active_win_rate = sum(r["active_wins"] for r in rows) / n
    mean_random_samples = float(np.mean([r["random_samples"] for r in rows]))
    mean_active_samples = float(np.mean([r["active_samples"] for r in rows]))
    reduction_pct = 100 * (1 - mean_active_samples / mean_random_samples)

    print(f"\n--- Active learning exploration ({n} tasks, controlled per §21, "
          f"init={FIXED_INIT.value} fixed) ---")
    print(f"active_win_rate={active_win_rate:.2f}  "
          f"mean_N_epsilon random={mean_random_samples:.0f} active={mean_active_samples:.0f}  "
          f"reduction={reduction_pct:+.1f}%")

    active_helps = active_win_rate > 0.5 and reduction_pct > 0
    record_gate_evaluation(
        generation="v1-active-learning", gate_number=0, metric_name="active_sampling_win_rate",
        metric_value=active_win_rate, threshold=0.5, n_samples=n,
        notes=f"mean_N_epsilon random={mean_random_samples:.0f} active={mean_active_samples:.0f} "
              f"reduction_pct={reduction_pct:+.1f} top_fraction={TOP_FRACTION} refresh_every={REFRESH_EVERY} "
              f"(docs.md §16 data-efficiency axis, source.md pillar 6, hard-example-mining heuristic)",
    )

    detail_rows = "\n".join(
        f"| {r['task']} | {r['random_steps']} | {r['active_steps']} | "
        f"{r['random_samples']} | {r['active_samples']} | {'active' if r['active_wins'] else 'random'} |"
        for r in rows
    )
    report = f"""## Method

Tests active sample selection (hard-example mining: periodically rescore
every training sample by current per-sample loss, restrict subsequent
batches to the hardest {TOP_FRACTION:.0%}, refreshed every {REFRESH_EVERY}
steps) against `precog.modes.train()`'s existing uniform-random batching --
source.md pillar 6 (Active Learning / Sample Efficiency), never tested in
this project before. Controlled per §21: architecture, task, learning rate
({FIXED_LEARNING_RATE}), batch size ({FIXED_BATCH_SIZE}), optimizer
({FIXED_OPTIMIZER}) and init ({FIXED_INIT.value}, the evidenced winner from
gate1_ranking.py) are all fixed and identical between arms across the same
{n} synthetic tasks used throughout this project. Since batch size is
identical in both arms, N_epsilon (samples to threshold, docs.md §16) is
just steps_to_threshold * batch_size, so a reduction in steps is a
reduction in N_epsilon by construction.

## Results

| task | random steps | active steps | random N_epsilon | active N_epsilon | winner |
|---|---:|---:|---:|---:|---|
{detail_rows}

**Active-sampling win rate: {active_win_rate:.0%}** ({sum(r['active_wins'] for r in rows)}/{n} tasks)
**Mean N_epsilon:** random={mean_random_samples:.0f}, active={mean_active_samples:.0f}
({reduction_pct:+.1f}% change, positive = active reduces samples needed)

## Verdict

Active sampling {'reduces' if active_helps else 'does NOT clearly reduce'}
N_epsilon on this benchmark ({'PASS' if active_helps else 'NEGATIVE RESULT'}
against the >50% win-rate + net-reduction bar). {"This is a genuinely new efficiency axis worth carrying into future work." if active_helps else "Consistent with the project's overall finding on these small, low-dimensional synthetic regression tasks: most sophistication tested so far (LSUV init, gradient_alignment proxy) has failed to beat simpler baselines -- the tasks may simply be too easy/low-dimensional for hard-example mining to matter, since even random batches already cover the input space densely at these sample sizes ({[t.n_samples for t in TASKS]})."}
"""
    export_csv_snapshots()
    report_path = write_report(
        "explore_active_learning", "Exploring Active Sample Selection (Pillar 6: Sample Efficiency)", report
    )
    print(f"\nReport written to {report_path.relative_to(report_path.parents[2])}")


if __name__ == "__main__":
    main()
