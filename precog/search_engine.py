"""Search Engine (docs.md §9.8): the Meta-Predictor supplies an informed
prior; this engine explores the remaining space. Optuna (already used as
the ground-truth generator, source.md pillar 1/8, and named directly in
§9.8/§18) plays the role of "exploration arm" here -- TPE's own density-
ratio sampler already balances exploration/exploitation (~ Expected
Improvement), so this wraps it with two things the raw sampler doesn't do
on its own:

  1. a warm start from the Meta-Predictor's recommendation (§9.7) instead
     of a cold, uninformed search -- the actual point of §9.8's diagram
     ("Meta Predictor -> Candidate Configs -> Rank/Optimize");
  2. a diversity penalty (docs.md's acquisition formula's gamma term)
     discouraging trials too close to ones already tried, complementing
     TPE's own exploration rather than duplicating it.

    Acquisition = alpha * ExpectedImprovement + beta * Uncertainty + gamma * Diversity

TPE's sampler already folds in something like the first two terms; only
diversity is added explicitly here, as a penalty subtracted from each
candidate's expected score before Optuna's own selection -- kept honest in
the report as an approximation of the full three-term formula, not a
literal implementation of it (docs.md §0's own disclaimer: targets and
formulas here are hypotheses to test, not established results).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import optuna

from precog.meta_predictor import MetaPredictor, Recommendation
from precog.model import InitMethod

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class SearchResult:
    best_learning_rate: float
    best_init: InitMethod
    best_steps: float
    trials_to_target: int | None  # None if the target was never reached
    n_trials: int
    diversity_weight: float
    history: list[dict] = field(default_factory=list)


class SearchEngine:
    """Wraps Optuna TPE with a Meta-Predictor warm start and a diversity
    penalty (docs.md §9.8's three-term acquisition, approximated)."""

    def __init__(self, diversity_weight: float = 0.1, seed: int = 0):
        self.diversity_weight = diversity_weight
        self.seed = seed

    def search(
        self,
        objective_fn,  # (learning_rate: float, init_method: InitMethod) -> steps_to_threshold (float)
        recommendation: Recommendation | None,
        n_trials: int,
        target_steps: float,
        lr_bounds: tuple[float, float] = (1e-4, 1e-1),
        fixed_init: InitMethod | None = None,
        log_run: tuple[str, int, str] | None = None,
    ) -> SearchResult:
        """`log_run`, when given as (run_label, task_seed, arm), records
        every trial via precog.experiment_db.record_search_trial (docs.md
        §12: "every experiment must be recorded" -- search trials included,
        kept in their own table since they aren't part of the meta-dataset's
        controlled train/test split, see experiment_db.py's schema comment)."""
        """`fixed_init`, when set, *restricts* the search to that single
        init_method instead of merely seeding one trial with it (docs.md
        §9.8's "Rank/Optimize" step acting on a narrowed candidate set) --
        every trial then goes toward refining learning_rate alone. This is
        the real test of whether trusting a recommendation pays off: it
        removes the 3x larger joint (LR, init) space for the trials that
        would otherwise be spent re-discovering init_method from scratch,
        at the risk of locking in a wrong choice if the recommendation is
        wrong (§27: a bad prior isn't free)."""
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed))

        if recommendation is not None and fixed_init is None:
            # §9.8: the Meta-Predictor's own top pick seeds the first trial
            # instead of the search starting cold (weak signal: one trial
            # among many, see results/reports/*_gate3_search_efficiency.md).
            study.enqueue_trial({"learning_rate": lr_bounds[0] * (lr_bounds[1] / lr_bounds[0]) ** 0.5,
                                  "init_method": recommendation.recommended_init.value})

        tried_points: list[tuple[float, str]] = []
        history = []
        trials_to_target = None

        def wrapped_objective(trial: optuna.Trial) -> float:
            nonlocal trials_to_target
            lr = trial.suggest_float("learning_rate", *lr_bounds, log=True)
            if fixed_init is not None:
                init_name = fixed_init.value
            else:
                init_name = trial.suggest_categorical("init_method", [m.value for m in InitMethod])
            steps = objective_fn(lr, InitMethod(init_name))

            diversity_penalty = 0.0
            if tried_points:
                log_lr = np.log10(lr)
                distances = [
                    abs(log_lr - np.log10(prev_lr)) + (0.0 if prev_init == init_name else 1.0)
                    for prev_lr, prev_init in tried_points
                ]
                diversity_penalty = self.diversity_weight * max(0.0, 1.0 - min(distances))
            tried_points.append((lr, init_name))

            score = steps + diversity_penalty * steps  # penalize near-duplicates, not free exploration
            history.append({"trial": trial.number, "learning_rate": lr, "init_method": init_name,
                             "steps": steps, "score": score})
            if log_run is not None:
                run_label, task_seed, arm = log_run
                from precog.experiment_db import record_search_trial

                record_search_trial(
                    run_label=run_label, seed=task_seed, arm=arm, trial_number=trial.number,
                    learning_rate=lr, init_method=init_name, steps_to_threshold=steps,
                )
            if trials_to_target is None and steps <= target_steps:
                trials_to_target = trial.number + 1
            return score

        study.optimize(wrapped_objective, n_trials=n_trials)

        best = study.best_params
        best_init = fixed_init if fixed_init is not None else InitMethod(best["init_method"])
        best_steps = min(h["steps"] for h in history if h["trial"] == study.best_trial.number)
        return SearchResult(
            best_learning_rate=best["learning_rate"],
            best_init=best_init,
            best_steps=best_steps,
            trials_to_target=trials_to_target,
            n_trials=n_trials,
            diversity_weight=self.diversity_weight,
            history=history,
        )
