"""Synthetic regression task generator (docs.md §15.2 "Laboratoire synthétique",
curriculum Niveau 1). Same generative functions as the earlier Rust prototype,
ported here so the whole V1 stack lives in one language (stack.md §1/§7:
PyTorch first, no Rust before the meta-predictor is validated)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch


class TaskFunction(str, Enum):
    LINEAR = "linear"
    NONLINEAR_INTERACTION = "nonlinear_interaction"
    NONLINEAR_PRODUCT = "nonlinear_product"


_MIN_INPUT_DIM = {
    TaskFunction.LINEAR: 2,
    TaskFunction.NONLINEAR_INTERACTION: 4,
    TaskFunction.NONLINEAR_PRODUCT: 3,
}


@dataclass
class TaskConfig:
    function: TaskFunction
    input_dim: int
    noise_level: float
    n_samples: int
    seed: int


def _eval_function(function: TaskFunction, x: np.ndarray) -> np.ndarray:
    if function == TaskFunction.LINEAR:
        return x[:, 0] + x[:, 1]
    if function == TaskFunction.NONLINEAR_INTERACTION:
        return np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2 - x[:, 2] * x[:, 3]
    if function == TaskFunction.NONLINEAR_PRODUCT:
        return np.sin(x[:, 0] * x[:, 1]) + np.exp(-x[:, 2])
    raise ValueError(function)


def generate(config: TaskConfig) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Returns (x, y, task_features). task_features are the only descriptors
    a PURE-mode data encoder (docs.md §9.2) is allowed to compute: no model
    ever sees more of the dataset than these statistics plus mini-batches."""
    if config.input_dim < _MIN_INPUT_DIM[config.function]:
        raise ValueError(
            f"input_dim too small for {config.function}: need >= {_MIN_INPUT_DIM[config.function]}"
        )
    rng = np.random.default_rng(config.seed)
    x = rng.standard_normal((config.n_samples, config.input_dim)).astype(np.float32)
    y = _eval_function(config.function, x)
    y = y + rng.standard_normal(config.n_samples).astype(np.float32) * config.noise_level

    corr = np.corrcoef(x, rowvar=False)
    off_diag = corr[np.triu_indices_from(corr, k=1)]
    # Redundancy (docs.md §9.2): fraction of feature pairs that are near-
    # duplicates, distinct from the mean correlation (a few highly
    # correlated pairs vs. many mildly correlated ones look the same under
    # a plain mean but very different under this threshold).
    redundancy = float(np.mean(np.abs(off_diag) > 0.9)) if off_diag.size else 0.0
    # Differential entropy of the target under a Gaussian approximation
    # (docs.md §9.2 "entropie") -- the task generator only ever produces
    # Gaussian-distributed inputs, so this closed form is exact for x and an
    # approximation for the (possibly non-Gaussian) transformed target y.
    target_variance = float(np.var(y))
    target_entropy_estimate = 0.5 * float(np.log(2 * np.pi * np.e * max(target_variance, 1e-12)))

    features = {
        "function": config.function.value,
        "seed": config.seed,
        "input_dim": config.input_dim,
        "noise_level": config.noise_level,
        "n_samples": config.n_samples,
        "target_variance": target_variance,
        "target_entropy_estimate": target_entropy_estimate,
        "feature_correlation_mean": float(np.mean(np.abs(off_diag))) if off_diag.size else 0.0,
        "redundancy": redundancy,
        "class_imbalance": None,  # not applicable: regression task, no classes
        "distribution": "gaussian",
    }
    return torch.from_numpy(x), torch.from_numpy(y).unsqueeze(1), features


def task_config_from_row(row) -> TaskConfig:
    """Reconstructs a TaskConfig from a meta-dataset row (precog.experiment_db
    .load_dataframe() output): generate() is a pure function of TaskConfig,
    so this losslessly reproduces the exact same synthetic (x, y) without
    storing the raw tensors themselves in the database."""
    return TaskConfig(
        function=TaskFunction(row["task.function"]),
        input_dim=int(row["task.input_dim"]),
        noise_level=float(row["task.noise_level"]),
        n_samples=int(row["task.n_samples"]),
        seed=int(row["task.seed"]),
    )
