"""Active Learning / data selection (docs.md §16 "Data efficiency", source.md
pillar 6): reduce N_epsilon, the number of samples needed to reach a target
performance -- a completely different axis from speed (T_epsilon), and one
this project hadn't touched at all until now.

Implements hard-example mining, the simplest well-established active-
learning heuristic that applies directly to a regression setting where
there's no natural "uncertainty" the way there is for classification
softmax entropy: periodically score every training sample by its current
per-sample loss, then bias subsequent batches toward the highest-loss
(hardest, most informative) examples instead of uniform random sampling.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from precog.model import InitMethod, ModelArchitecture, build_mlp
from precog.modes import TrainingConfig, TrainProtocol


@dataclass
class ActiveTrainResult:
    initial_loss: float
    final_loss: float
    steps_to_threshold: int | None
    samples_seen_to_threshold: int | None  # N_epsilon (docs.md §16/§17)
    converged: bool
    diverged: bool
    wall_clock_s: float


def _build_optimizer(params, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.learning_rate)
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate, weight_decay=0.0)
    return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)


def train_active(
    architecture: ModelArchitecture,
    x: torch.Tensor,
    y: torch.Tensor,
    training: TrainingConfig,
    protocol: TrainProtocol,
    top_fraction: float = 0.5,
    refresh_every: int = 8,
) -> ActiveTrainResult:
    """Every `refresh_every` steps, rescores all samples by current
    per-sample loss and restricts subsequent batches to the hardest
    `top_fraction` of them -- everything else about the loop mirrors
    precog.modes.train() exactly, so the only difference under test is the
    sample-selection policy."""
    torch.manual_seed(protocol.seed)
    generator = torch.Generator().manual_seed(protocol.seed)

    model = build_mlp(architecture, training.init_method)
    optimizer = _build_optimizer(model.parameters(), training)
    loss_fn = nn.functional.mse_loss

    n_samples = x.shape[0]
    batch_size = min(training.batch_size, n_samples)
    pool_size = max(batch_size, int(n_samples * top_fraction))

    steps_to_threshold = None
    samples_seen_to_threshold = None
    initial_loss = float("nan")
    final_loss = float("nan")
    diverged = False
    samples_seen = 0
    hard_pool = torch.arange(n_samples)

    start = time.perf_counter()
    for step in range(protocol.max_steps):
        if step % refresh_every == 0:
            with torch.no_grad():
                per_sample_loss = ((model(x) - y) ** 2).squeeze(-1)
            hard_pool = torch.topk(per_sample_loss, pool_size).indices

        batch_positions = torch.randint(0, len(hard_pool), (batch_size,), generator=generator)
        batch_idx = hard_pool[batch_positions]

        pred = model(x[batch_idx])
        loss = loss_fn(pred, y[batch_idx])
        loss_value = loss.item()
        samples_seen += batch_size

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
            samples_seen_to_threshold = samples_seen
            break

    wall_clock_s = time.perf_counter() - start
    return ActiveTrainResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        steps_to_threshold=steps_to_threshold,
        samples_seen_to_threshold=samples_seen_to_threshold,
        converged=steps_to_threshold is not None,
        diverged=diverged,
        wall_clock_s=wall_clock_s,
    )
