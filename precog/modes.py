"""The Zero-Training Contract (docs.md §5) -- the most important
methodological constraint in the project, enforced here as code, not just
convention.

  PURE          ΔW = 0   -- never trains. There is no `Mode.PURE` branch in
                            train() below: PURE-mode analysis lives entirely
                            in trainability.py/encoders and never touches
                            this module, by construction.
  PROBE         ΔW != 0, bounded and logged -- a short, explicitly budgeted
                            training run used to refine a PURE prediction.
  FULL_TRAINING ΔW != 0, unrestricted -- ground-truth generation. Never used
                            to "cheat" on a PURE or PROBE prediction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from precog.model import ModelArchitecture, InitMethod, build_mlp


class Mode(str, Enum):
    PROBE = "probe"
    FULL_TRAINING = "full_training"


@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    optimizer: str  # "sgd" | "adam" | "adamw"
    weight_decay: float
    init_method: InitMethod


@dataclass
class TrainProtocol:
    mode: Mode
    max_steps: int
    loss_threshold: float
    seed: int


@dataclass
class TrainResult:
    mode: Mode
    initial_loss: float
    final_loss: float
    steps_to_threshold: int | None
    converged: bool
    diverged: bool
    wall_clock_s: float
    delta_w_norm: float  # ||W_final - W_initial||, logged per the ΔW contract


def _build_optimizer(params, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.learning_rate)
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate, weight_decay=0.0)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"unknown optimizer {config.optimizer!r}")


def train(
    architecture: ModelArchitecture,
    x: torch.Tensor,
    y: torch.Tensor,
    training: TrainingConfig,
    protocol: TrainProtocol,
) -> TrainResult:
    """PROBE or FULL_TRAINING only. Every call here means DeltaW != 0 by
    contract -- if you want PURE-mode analysis, use precog.trainability
    instead, which never imports this module's optimizer step."""
    if protocol.mode == Mode.PROBE and not (50 <= protocol.max_steps <= 1000):
        raise ValueError(
            f"PROBE is bounded by contract (docs.md §5: 'e.g. 50-1000 steps, "
            f"0.1-1% of the total budget'), got max_steps={protocol.max_steps}. "
            f"Use Mode.FULL_TRAINING if you actually need an unrestricted run."
        )

    torch.manual_seed(protocol.seed)
    generator = torch.Generator().manual_seed(protocol.seed)

    model = build_mlp(architecture, training.init_method)
    initial_weights = torch.cat([p.detach().flatten() for p in model.parameters()])

    optimizer = _build_optimizer(model.parameters(), training)
    loss_fn = nn.functional.mse_loss

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
        batch_idx = order[(step * batch_size) % n_samples : (step * batch_size) % n_samples + batch_size]
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
        mode=protocol.mode,
        initial_loss=initial_loss,
        final_loss=final_loss,
        steps_to_threshold=steps_to_threshold,
        converged=steps_to_threshold is not None,
        diverged=diverged,
        wall_clock_s=wall_clock_s,
        delta_w_norm=float((final_weights - initial_weights).norm().item()),
    )
