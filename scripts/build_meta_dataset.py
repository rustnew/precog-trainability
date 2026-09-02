#!/usr/bin/env python3
"""Populates the meta-dataset (docs.md §12) for the first real Meta-Predictor
(§9.7): for a batch of synthetic tasks, runs the same controlled init-only
experiment as gate1_ranking.py (§21: optimizer/LR/batch fixed, only
init_method varies) and records every trial with its PURE-mode zero-cost
features and its FULL_TRAINING ground truth.

Strict TRAIN/VALIDATION/TEST separation (§12, §15.1): the split is assigned
by *task* (not by trial) at generation time, and TEST is locked -- never
read by scripts/train_meta_predictor.py during development, only for a
final, one-shot evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random

import torch

from precog.experiment_db import experiment_exists, record_experiment
from precog.hardware import hardware_features
from precog.model import Activation, InitMethod, ModelArchitecture, build_mlp, model_features
from precog.modes import Mode, TrainingConfig, TrainProtocol, train
from precog.regime import detect_regime
from precog.taskgen import TaskConfig, TaskFunction, generate
from precog.trainability import zero_cost_features

FIXED_OPTIMIZER = "adam"
FIXED_LEARNING_RATE = 0.02
FIXED_BATCH_SIZE = 32
FULL_MAX_STEPS = 800
FULL_LOSS_THRESHOLD = 0.05
INIT_METHODS = [InitMethod.XAVIER, InitMethod.HE, InitMethod.ORTHOGONAL]
FUNCTIONS = [TaskFunction.LINEAR, TaskFunction.NONLINEAR_INTERACTION, TaskFunction.NONLINEAR_PRODUCT]


def build_tasks(n_tasks: int, seed_offset: int = 100) -> list[TaskConfig]:
    tasks = []
    for i in range(n_tasks):
        rng = random.Random(seed_offset + i)
        function = FUNCTIONS[i % len(FUNCTIONS)]
        min_dim = {TaskFunction.LINEAR: 2, TaskFunction.NONLINEAR_INTERACTION: 4, TaskFunction.NONLINEAR_PRODUCT: 3}[function]
        tasks.append(
            TaskConfig(
                function=function,
                input_dim=rng.randint(min_dim, min_dim + 4),
                noise_level=round(rng.uniform(0.0, 0.3), 3),
                n_samples=rng.choice([256, 384, 512, 768]),
                seed=seed_offset + i,
            )
        )
    return tasks


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, default=80)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    tasks = build_tasks(args.n_tasks)
    n_skipped, n_logged = 0, 0

    for task_idx, task_config in enumerate(tasks):
        # Split assignment is a deterministic function of the task's own
        # seed, not of its position in `tasks` -- growing --n-tasks must
        # never re-shuffle a task that was already assigned to TRAIN into
        # TEST (or vice versa), which a range-based rng.sample() would do
        # silently every time len(tasks) changes (found the hard way: it
        # would have broken §15.1's "TEST is locked" guarantee on every
        # meta-dataset scale-up).
        split = "test" if random.Random(f"split-{task_config.seed}").random() < args.test_fraction else "train"
        x, y, task_feat = generate(task_config)

        for init_method in INIT_METHODS:
            if experiment_exists(task_config.seed, init_method.value):
                n_skipped += 1
                continue
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

            torch.manual_seed(0)
            pure_model = build_mlp(architecture, init_method)
            zc = zero_cost_features(pure_model, task_config.input_dim, x, y)
            model_feat = model_features(pure_model, architecture, init_method)
            hw_feat = hardware_features()
            regime = detect_regime(model_feat, task_feat, hw_feat)

            protocol = TrainProtocol(
                mode=Mode.FULL_TRAINING, max_steps=FULL_MAX_STEPS, loss_threshold=FULL_LOSS_THRESHOLD, seed=0
            )
            result = train(architecture, x, y, training, protocol)

            record_experiment(
                split=split,
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
            n_logged += 1

        print(f"[{task_idx + 1}/{len(tasks)}] split={split} task={task_config.function.value} logged")

    print(f"done. {n_logged} new experiments logged, {n_skipped} already present and skipped.")


if __name__ == "__main__":
    main()
