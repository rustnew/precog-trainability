#!/usr/bin/env python3
"""What PRECOG can recommend *today*, end to end (docs.md §29 Production
Architecture, in miniature): given a task, run every V1 pipeline stage in
order (Model/Data/Hardware Encoders -> Trainability Engine -> Regime
Detector -> Meta-Knowledge Base -> Meta-Predictor -> Search Engine) and
print the final recommended configuration in §9.7's own output format
(a distribution + confidence, never a bare point value) -- with an explicit,
honest note on which of the 4 V1 target hyperparameters (docs.md §25:
Learning Rate, Batch Size, Optimizer, Initialization) are actually learned
vs. still fixed by convention at this stage.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from precog.experiment_db import load_dataframe
from precog.hardware import hardware_features
from precog.meta_knowledge_base import MetaKnowledgeBase
from precog.meta_predictor import (
    REDUCED_FEATURE_COLUMNS,
    MetaPredictor,
    compute_candidate_zero_cost,
    engineer_features,
)
from precog.model import Activation, InitMethod, ModelArchitecture, model_features, build_mlp
from precog.modes import Mode, TrainingConfig, TrainProtocol, train
from precog.regime import detect_regime
from precog.search_engine import SearchEngine
from precog.taskgen import TaskConfig, TaskFunction, generate

NON_CONVERGENCE_PENALTY = 800 * 2
FIXED_OPTIMIZER = "adam"  # not yet predicted -- see closing note
FIXED_BATCH_SIZE = 32  # not yet predicted -- see closing note
FULL_MAX_STEPS = 800
FULL_LOSS_THRESHOLD = 0.05
N_SEARCH_TRIALS = 15


def recommend_for_task(task_config: TaskConfig, architecture: ModelArchitecture) -> None:
    x, y, task_feat = generate(task_config)

    # Model/Hardware/Regime Encoders (§9.1/§9.3/§9.5) -- computed once,
    # init-independent parts only (n_params/flops/depth/width don't depend
    # on which init is chosen).
    torch_model = build_mlp(architecture, InitMethod.XAVIER)  # placeholder init just to read architecture stats
    model_feat = model_features(torch_model, architecture, InitMethod.XAVIER)
    hw_feat = hardware_features()
    regime = detect_regime(model_feat, task_feat, hw_feat)

    print("=== Model/Data/Hardware/Regime (§9.1/§9.2/§9.3/§9.5) ===")
    print(f"task: function={task_config.function.value} input_dim={task_config.input_dim} "
          f"noise={task_config.noise_level} n_samples={task_config.n_samples}")
    print(f"model: depth={model_feat['depth']} width={model_feat['width']} n_params={model_feat['n_params']}")
    print(f"regime: {regime['regime_label']}")

    # Meta-Knowledge Base + Meta-Predictor (§9.6/§9.7): trained on the
    # locked TRAIN split, the evidenced-best design (reduced_rf, see
    # results/reports/*_compare_meta_predictors.md).
    train_df = load_dataframe(split="train")
    train_df["steps_to_threshold"] = train_df["steps_to_threshold"].fillna(NON_CONVERGENCE_PENALTY)
    mkb = MetaKnowledgeBase(k=5)
    mkb.fit(train_df)
    engineered_rows = [engineer_features(row.to_frame().T, mkb) for _, row in train_df.iterrows()]
    predictor = MetaPredictor(feature_columns=REDUCED_FEATURE_COLUMNS)
    predictor.fit(pd.concat(engineered_rows, ignore_index=True), train_df["training.init_method"],
                  train_df["steps_to_threshold"])

    features_row = pd.DataFrame([{**{f"task.{k}": v for k, v in task_feat.items()},
                                   **{f"model.{k}": v for k, v in model_feat.items()}}])
    engineered = engineer_features(features_row, mkb)
    zc_by_candidate = compute_candidate_zero_cost(architecture, task_config.input_dim, x, y)
    recommendation = predictor.recommend(engineered, zc_by_candidate)

    print("\n=== Meta-Predictor recommendation (§9.7) -- a distribution, not a point value ===")
    for candidate, pred in recommendation.per_candidate.items():
        marker = " <-- recommended" if candidate == recommendation.recommended_init.value else ""
        print(f"  init={candidate:<11} expected_steps={pred['expected_steps']:.0f} "
              f"+/-{pred['std_steps']:.0f}{marker}")
    print(f"confidence: {recommendation.confidence:.0%} "
          f"(calibration note: reduced_rf measured ~44% real accuracy on the locked test split, "
          f"see results/reports/*_compare_meta_predictors.md -- treat this confidence as indicative, not exact)")

    # Search Engine (§9.8): joint (LR, init) search is the evidenced-better
    # strategy (gate3b showed restricting to the recommendation underperforms
    # a cold joint search) -- so the recommendation biases the starting
    # point but does not remove init_method from the search.
    def objective_fn(lr: float, init_method: InitMethod) -> float:
        training = TrainingConfig(learning_rate=lr, batch_size=FIXED_BATCH_SIZE, optimizer=FIXED_OPTIMIZER,
                                   weight_decay=1e-5, init_method=init_method)
        protocol = TrainProtocol(mode=Mode.FULL_TRAINING, max_steps=FULL_MAX_STEPS,
                                  loss_threshold=FULL_LOSS_THRESHOLD, seed=0)
        result = train(architecture, x, y, training, protocol)
        return result.steps_to_threshold if result.converged else NON_CONVERGENCE_PENALTY

    search_result = SearchEngine(seed=0).search(
        objective_fn, recommendation, N_SEARCH_TRIALS, target_steps=100.0,
        log_run=("predict", task_config.seed, "demo"),
    )

    print(f"\n=== Search Engine result (§9.8, {N_SEARCH_TRIALS} trials, joint LR x init) ===")
    print(f"best found: learning_rate={search_result.best_learning_rate:.4g} "
          f"init={search_result.best_init.value} -> steps_to_threshold={search_result.best_steps:.0f}")

    print("\n=== Final recommended configuration ===")
    print(f"  learning_rate : {search_result.best_learning_rate:.4g}   (found by search, not yet predicted directly)")
    print(f"  batch_size    : {FIXED_BATCH_SIZE}   (FIXED by convention -- not yet predicted or searched, see note)")
    print(f"  optimizer     : {FIXED_OPTIMIZER}   (FIXED by convention -- not yet predicted or searched, see note)")
    print(f"  init_method   : {search_result.best_init.value}   (Meta-Predictor recommended "
          f"'{recommendation.recommended_init.value}'; search confirmed/corrected it)")
    print(f"  expected steps_to_threshold: {search_result.best_steps:.0f}")

    print(
        "\nNote on V1 scope (docs.md §25 names 4 targets: Learning Rate, Batch Size, Optimizer,\n"
        "Initialization): only *initialization* has a learned, task-conditional predictor today\n"
        "(44% test accuracy, below the 56% universal baseline -- H4 not yet refuted, see §17\n"
        "Gate 2). Learning rate is *discovered* by the Search Engine (§9.8), not predicted by the\n"
        "Meta-Predictor. Batch size and optimizer are still fixed by convention across every\n"
        "experiment so far -- neither predicted nor searched. This output is best read as 'what\n"
        "the current V1 pipeline can already automate end-to-end', not as a validated\n"
        "recommendation to trust blindly."
    )

    from precog.reporting import export_csv_snapshots, write_report

    per_candidate_table = "\n".join(
        f"| {c} | {p['expected_steps']:.0f} | {p['std_steps']:.0f} |"
        f"{' (recommended)' if c == recommendation.recommended_init.value else ''}"
        for c, p in recommendation.per_candidate.items()
    )
    report = f"""## Task

function={task_config.function.value}, input_dim={task_config.input_dim}, noise={task_config.noise_level},
n_samples={task_config.n_samples}, seed={task_config.seed} (not in TRAIN or TEST -- a genuinely new task).
Model: depth={model_feat['depth']} width={model_feat['width']} n_params={model_feat['n_params']}.
Regime (§9.5): {regime['regime_label']}.

## Meta-Predictor recommendation (§9.7)

| init | expected steps | +/- std | |
|---|---:|---:|---|
{per_candidate_table}

Confidence: {recommendation.confidence:.0%}.

## Search Engine result (§9.8, {N_SEARCH_TRIALS} trials, joint LR x init)

best found: learning_rate={search_result.best_learning_rate:.4g}, init={search_result.best_init.value}
-> steps_to_threshold={search_result.best_steps:.0f}

## Final recommended configuration

| hyperparameter | value | status |
|---|---|---|
| learning_rate | {search_result.best_learning_rate:.4g} | found by search, not yet predicted directly |
| batch_size | {FIXED_BATCH_SIZE} | FIXED by convention -- not yet predicted or searched |
| optimizer | {FIXED_OPTIMIZER} | FIXED by convention -- not yet predicted or searched |
| init_method | {search_result.best_init.value} | Meta-Predictor recommended '{recommendation.recommended_init.value}'; search {'confirmed' if search_result.best_init == recommendation.recommended_init else 'overrode'} it |

Expected steps_to_threshold: {search_result.best_steps:.0f}

## Honest caveat

Per docs.md §25's 4 V1 targets (Learning Rate, Batch Size, Optimizer,
Initialization), only *initialization* has a learned, task-conditional
predictor today, and it measured ~44% test accuracy (below the 56%
universal baseline -- §17 Gate 2 not met, H4 not refuted). Learning rate is
*discovered* by the Search Engine, not predicted. Batch size and optimizer
remain fixed by convention, neither predicted nor searched. This report
shows what the V1 pipeline can already automate end to end -- not a
validated recommendation.
"""
    export_csv_snapshots()
    report_path = write_report("predict_demo", "End-to-End Recommendation Demo", report)
    print(f"\nReport written to {report_path.relative_to(report_path.parents[2])}")


if __name__ == "__main__":
    task_config = TaskConfig(
        function=TaskFunction.NONLINEAR_INTERACTION, input_dim=6, noise_level=0.15, n_samples=512, seed=9999
    )
    architecture = ModelArchitecture(input_dim=task_config.input_dim, depth=2, width=32, activation=Activation.RELU)
    recommend_for_task(task_config, architecture)
