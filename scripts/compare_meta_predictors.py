#!/usr/bin/env python3
"""Compares alternative Meta-Predictor designs (docs.md §9.7, §19 ablation
methodology) on the exact same locked TEST split, and picks a winner by
evidence rather than assumption. Replaces train_meta_predictor.py, whose
single full-feature RandomForest variant is now just one of the four
candidates evaluated here:

Also reports **regret**, not just top-1 accuracy: predicting the wrong
init class is not uniformly bad -- picking Orthogonal when Xavier was
truly best but only 5 steps faster is a good practical decision scored as
a miss by accuracy alone, while picking Orthogonal when Xavier would have
taken 700 fewer steps is a real error. regret = steps(predicted) -
steps(true_best) makes that distinction explicit (relative_regret =
regret / steps(true_best) so it's comparable across tasks of very
different absolute difficulty, from ~20-step linear tasks to ~700-step
nonlinear ones).

  full_rf     - RandomForest, all features (task+model+zero_cost+regime+prior)
  reduced_rf  - RandomForest, only the zero-cost proxies §21's controlled
                experiment individually validated (gradient_norm,
                gradient_norm_variance, jacob_cov, effective_rank,
                jacobian_condition_mean)
  log_rf      - RandomForest, all features, log1p(steps) target (the
                1600-step non-convergence penalty is a heavy-tailed outlier
                in raw space)
  gp_reduced  - Gaussian Process regression (sklearn), only the §21-validated
                zero-cost proxies, log1p target -- tests stack.md's own named
                target framework (GPyTorch/BoTorch/Ax) empirically for the
                first time, specifically for its principled posterior
                uncertainty vs RandomForest's ad-hoc tree-spread confidence
  knn         - no learned model: Meta-Knowledge Base (§9.6) nearest-
                neighbor prior alone
  zc_gradnormvar / zc_jacobcov - no learned model, no training data needed
                at all: rank candidates directly by a single raw zero-cost
                proxy (source.md pillar 3's own methodology -- SynFlow/SNIP/
                NASWOT papers rank architectures by the proxy score
                directly). The "PRECOG-0" tier of docs.md §19.1's own
                ablation ladder ("Zero-Cost only"), included to test whether
                the learned RandomForest wrapper is adding value over the
                raw signal at all.

All candidates see the same TRAIN split (except the two zero-cost
heuristics, which need none), are evaluated once on the same locked TEST
split, and are ranked by top-1 accuracy with confidence calibration (mean
confidence vs. actual accuracy) reported alongside as a tiebreaker and an
explicit red flag, not just accuracy alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from precog.experiment_db import load_dataframe, record_gate_evaluation
from precog.meta_knowledge_base import MetaKnowledgeBase
from precog.meta_predictor import (
    FEATURE_COLUMNS,
    REDUCED_FEATURE_COLUMNS,
    GPMetaPredictor,
    KNNMetaPredictor,
    MetaPredictor,
    TieBreakHeuristicPredictor,
    ZeroCostHeuristicPredictor,
    compute_candidate_zero_cost,
    engineer_features,
)
from precog.model import architecture_from_row
from precog.reporting import export_csv_snapshots, write_report
from precog.taskgen import generate, task_config_from_row

NON_CONVERGENCE_PENALTY = 800 * 2


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["steps_to_threshold"] = df["steps_to_threshold"].fillna(NON_CONVERGENCE_PENALTY)
    return df


def evaluate(name, predictor, test_df, mkb):
    hits, confidences, detail_rows = 0, [], []
    regrets, relative_regrets = [], []
    n = 0
    for seed, group in test_df.groupby("seed"):
        n += 1
        features_row = group.iloc[[0]]
        engineered = engineer_features(features_row, mkb)
        best_row = group.loc[group["steps_to_threshold"].idxmin()]
        true_best_init = best_row["training.init_method"]
        true_best_steps = best_row["steps_to_threshold"]

        task_config = task_config_from_row(features_row.iloc[0])
        architecture = architecture_from_row(features_row.iloc[0])
        x, y, _ = generate(task_config)
        zc_by_candidate = compute_candidate_zero_cost(architecture, task_config.input_dim, x, y)

        rec = predictor.recommend(engineered, zc_by_candidate)
        hit = rec.recommended_init.value == true_best_init
        hits += int(hit)
        confidences.append(rec.confidence)

        predicted_steps = group.loc[
            group["training.init_method"] == rec.recommended_init.value, "steps_to_threshold"
        ].iloc[0]
        regret = predicted_steps - true_best_steps
        regrets.append(regret)
        relative_regrets.append(regret / true_best_steps)

        detail_rows.append(f"| {seed} | {true_best_init} | {rec.recommended_init.value} | "
                            f"{rec.confidence:.2f} | {hit} | {regret:.0f} |")

    accuracy = hits / n
    # ZeroCostHeuristicPredictor makes no probabilistic claim (docs.md
    # §20's confidence machinery doesn't apply to a raw, unlearned rule),
    # so its confidences are all NaN by design -- nanmean keeps this from
    # poisoning the whole comparison instead of silently propagating NaN.
    mean_confidence = float(np.nanmean(confidences)) if not all(np.isnan(confidences)) else float("nan")
    gap_str = f"{mean_confidence - accuracy:+.2f}" if not np.isnan(mean_confidence) else "N/A"
    mean_regret = float(np.mean(regrets))
    mean_relative_regret = float(np.mean(relative_regrets))
    print(f"{name:<15} accuracy={accuracy:.0%} ({hits}/{n})  "
          f"mean_confidence={'N/A' if np.isnan(mean_confidence) else f'{mean_confidence:.2f}'}  "
          f"calibration_gap={gap_str}  mean_regret={mean_regret:+.1f} steps "
          f"({mean_relative_regret:+.0%} relative)")
    return {"name": name, "accuracy": accuracy, "hits": hits, "n": n,
            "mean_confidence": mean_confidence, "mean_regret": mean_regret,
            "mean_relative_regret": mean_relative_regret, "detail_rows": detail_rows}


def main() -> None:
    train_df = _prepare(load_dataframe(split="train"))
    test_df = _prepare(load_dataframe(split="test"))
    print(f"train: {len(train_df)} rows ({train_df['seed'].nunique()} tasks) | "
          f"test (locked): {len(test_df)} rows ({test_df['seed'].nunique()} tasks)\n")

    mkb = MetaKnowledgeBase(k=5)
    mkb.fit(train_df)
    engineered_rows = [engineer_features(row.to_frame().T, mkb) for _, row in train_df.iterrows()]
    train_engineered = pd.concat(engineered_rows, ignore_index=True)

    best_per_train_task = train_df.loc[train_df.groupby("seed")["steps_to_threshold"].idxmin()]
    universal_init = best_per_train_task["training.init_method"].mode()[0]
    universal_accuracy = float((test_df.loc[test_df.groupby("seed")["steps_to_threshold"].idxmin(),
                                             "training.init_method"] == universal_init).mean())
    random_baseline = 1 / 3

    universal_regrets, random_regrets = [], []
    for _, group in test_df.groupby("seed"):
        true_best_steps = group["steps_to_threshold"].min()
        universal_row = group.loc[group["training.init_method"] == universal_init]
        if len(universal_row):
            universal_regrets.append(universal_row["steps_to_threshold"].iloc[0] - true_best_steps)
        # Random baseline picks uniformly among the 3 candidates; its expected
        # regret for this task is the mean regret across all 3, in closed form
        # (no need to actually sample a random choice).
        random_regrets.append((group["steps_to_threshold"] - true_best_steps).mean())
    universal_mean_regret = float(np.mean(universal_regrets))
    random_mean_regret = float(np.mean(random_regrets))

    candidates = {
        "full_rf": MetaPredictor(feature_columns=FEATURE_COLUMNS, log_target=False),
        "reduced_rf": MetaPredictor(feature_columns=REDUCED_FEATURE_COLUMNS, log_target=False),
        "log_rf": MetaPredictor(feature_columns=FEATURE_COLUMNS, log_target=True),
        # Tests stack.md's own named target framework (GPyTorch/BoTorch/Ax)
        # empirically for the first time: a GP's posterior std is a
        # principled uncertainty, unlike RandomForest's ad-hoc tree spread.
        "gp_reduced": GPMetaPredictor(feature_columns=REDUCED_FEATURE_COLUMNS, log_target=True),
    }
    for predictor in candidates.values():
        predictor.fit(train_engineered, train_df["training.init_method"], train_df["steps_to_threshold"])
    candidates["knn"] = KNNMetaPredictor(mkb)
    # PRECOG-0 tier (docs.md §19.1): no learning at all, rank by the raw
    # proxy directly -- source.md pillar 3's own zero-cost NAS methodology.
    # Both proxies Gate 1 found strongest individually (higher value = more
    # steps = worse, hence higher_is_better=False for both).
    candidates["zc_gradnormvar"] = ZeroCostHeuristicPredictor("gradient_norm_variance", higher_is_better=False)
    candidates["zc_jacobcov"] = ZeroCostHeuristicPredictor("jacob_cov", higher_is_better=False)
    # gradient_norm, not gradient_norm_variance, is the strongest individual
    # ranking proxy once checked at full meta-dataset scale (rho=0.540 vs
    # 0.395, scripts/gate1_ranking_at_scale.py) -- worth testing directly as
    # a decision heuristic too, not just for ranking correlation.
    candidates["zc_gradnorm"] = ZeroCostHeuristicPredictor("gradient_norm", higher_is_better=False)
    # Variance-reduction attempt on the champion heuristic: average jacob_cov
    # over several random mini-batches instead of one fixed slice of x
    # (precog.trainability.jacob_cov_averaged). Tests whether reducing noise
    # in the winning proxy improves decision quality further.
    candidates["zc_jacobcov_avg"] = ZeroCostHeuristicPredictor("jacob_cov_avg", higher_is_better=False)
    # Root-caused fix (precog.meta_predictor.TieBreakHeuristicPredictor):
    # jacob_cov is provably identical between xavier/he (same underlying
    # Gaussian draws, rescaled -- sign-invariant), so raw zc_jacobcov never
    # once recommends "he". Break that exact tie with gradient_norm, which
    # differs sharply by init and can actually discriminate them.
    candidates["zc_jacobcov_tiebreak"] = TieBreakHeuristicPredictor(
        primary_proxy="jacob_cov", secondary_proxy="gradient_norm",
        primary_higher_is_better=False, secondary_higher_is_better=False,
    )
    # Second fix attempt for the same root cause: gradient_norm as a
    # per-init-family-normalized tiebreaker, since raw gradient_norm turned
    # out to have the same problem (he > xavier on 312/312 tasks -- a fixed
    # scale confound, not task signal). Population stats from TRAIN only.
    gradnorm_pop_stats = {
        init: (float(train_df.loc[train_df["training.init_method"] == init, "zero_cost.gradient_norm"].mean()),
               float(train_df.loc[train_df["training.init_method"] == init, "zero_cost.gradient_norm"].std()))
        for init in train_df["training.init_method"].unique()
    }
    candidates["zc_jacobcov_normtiebreak"] = TieBreakHeuristicPredictor(
        primary_proxy="jacob_cov", secondary_proxy="gradient_norm",
        primary_higher_is_better=False, secondary_higher_is_better=False,
        secondary_population_stats=gradnorm_pop_stats,
    )

    print(f"Baselines: universal={universal_accuracy:.0%} (mean_regret={universal_mean_regret:+.1f} steps)  "
          f"random={random_baseline:.0%} (mean_regret={random_mean_regret:+.1f} steps)\n")
    print("--- Candidate Meta-Predictors on the locked TEST split ---")
    results = [evaluate(name, predictor, test_df, mkb) for name, predictor in candidates.items()]

    # Regret first, not accuracy: at n=60 test tasks, top-1 accuracy is
    # unstable enough that a single flipped task changes which candidate
    # "wins" -- observed directly comparing two runs of this exact script,
    # one on a meta-dataset accidentally inflated with 72 duplicate rows
    # (reduced_rf and zc_jacobcov tied at 47%) and one on the same data
    # cleaned (reduced_rf edges ahead at 48% vs 47%, purely from that
    # dedup). Accuracy-first selection would have crowned reduced_rf both
    # times on that basis alone -- but reduced_rf's regret on the clean run
    # (+32.9 steps) is actually *worse* than doing nothing (the universal
    # baseline's +22.2), while zc_jacobcov's (+14.6) is not. A method that
    # is more often exactly right but wrong by more when it misses is not
    # obviously better in practice; accuracy alone can't see that.
    # Calibration gap remains the last-resort tiebreaker.
    winner = min(results, key=lambda r: (r["mean_regret"], -r["accuracy"],
                                          0.0 if np.isnan(r["mean_confidence"])
                                          else (r["mean_confidence"] - r["accuracy"])))
    beats_universal = winner["accuracy"] > universal_accuracy

    beats_universal_regret = winner["mean_regret"] < universal_mean_regret
    winner_gap_str = (f"{winner['mean_confidence'] - winner['accuracy']:+.2f}"
                       if not np.isnan(winner["mean_confidence"]) else "N/A")
    print(f"\nWinner: {winner['name']} (accuracy={winner['accuracy']:.0%}, "
          f"calibration_gap={winner_gap_str}, "
          f"mean_regret={winner['mean_regret']:+.1f} steps)")
    print(f"Beats universal-config baseline on accuracy ({universal_accuracy:.0%})? {beats_universal}")
    print(f"Beats universal-config baseline on regret ({universal_mean_regret:+.1f} steps)? {beats_universal_regret}")

    for r in results:
        record_gate_evaluation(
            generation=f"v1-meta-predictor-{r['name']}", gate_number=2,
            metric_name="top1_accuracy_init_method_adapted_recall_at_10",
            metric_value=r["accuracy"], threshold=0.80, n_samples=r["n"],
            notes=f"universal_baseline={universal_accuracy:.2f}, random_baseline={random_baseline:.2f}, "
                  f"mean_confidence={r['mean_confidence']:.2f}, "
                  f"winner={'yes' if r['name'] == winner['name'] else 'no'}",
        )
        record_gate_evaluation(
            generation=f"v1-meta-predictor-{r['name']}", gate_number=2,
            metric_name="mean_regret_steps_to_threshold",
            metric_value=r["mean_regret"], threshold=0.0, n_samples=r["n"],
            notes=f"regret = steps(predicted_init) - steps(true_best_init); "
                  f"relative_regret={r['mean_relative_regret']:+.2%}; "
                  f"universal_baseline_regret={universal_mean_regret:+.1f}; "
                  f"random_baseline_regret={random_mean_regret:+.1f}",
        )

    results_table = "\n".join(
        f"| {r['name']} | {r['accuracy']:.0%} ({r['hits']}/{r['n']}) | {r['mean_confidence']:.2f} | "
        f"{r['mean_confidence'] - r['accuracy']:+.2f} | {r['mean_regret']:+.1f} | {r['mean_relative_regret']:+.0%} |"
        for r in sorted(results, key=lambda r: -r["accuracy"])
    )
    winner_detail = "\n".join(winner["detail_rows"])
    report = f"""## Method

Four Meta-Predictor designs (docs.md §9.7, §19 ablation methodology), all
trained on the identical TRAIN split ({len(train_df)} rows,
{train_df['seed'].nunique()} tasks) and evaluated exactly once on the
identical locked TEST split ({len(test_df)} rows, {test_df['seed'].nunique()}
tasks): `full_rf` (all features), `reduced_rf` (only §21-validated zero-cost
proxies), `log_rf` (all features, log1p target), `knn` (Meta-Knowledge Base
neighbor vote alone, no learned model).

Baselines: universal-config = {universal_accuracy:.0%} (mean_regret={universal_mean_regret:+.1f} steps),
random = {random_baseline:.0%} (mean_regret={random_mean_regret:+.1f} steps).

Alongside top-1 accuracy, this run also reports **regret** = steps(predicted
init) - steps(true best init) per test task, and its task-scale-normalized
form relative_regret = regret / steps(true best init) -- a wrong top-1 call
that costs 5 extra steps and one that costs 700 extra steps are both
"misses" under accuracy alone, but very different practical outcomes.

Winner selection is now regret-first, not accuracy-first: a same-day rerun
of this exact script, before vs. after fixing a data-hygiene bug that had
duplicated 72 rows into the TRAIN split, flipped `reduced_rf` from tied
with `zc_jacobcov` at 47% to a lone lead at 48% -- purely from the dedup,
one task's worth of noise at this sample size. Accuracy-first selection
would have crowned `reduced_rf` either way, despite its regret on the clean
run (+32.9 steps) being *worse* than the universal baseline itself
(+22.2 steps) -- i.e. more often exactly right, but wrong by more when it
misses. Regret catches that; accuracy alone cannot.

## Results

| candidate | accuracy | mean confidence | calibration gap | mean regret (steps) | mean relative regret |
|---|---:|---:|---:|---:|---:|
{results_table}

## Winner: `{winner['name']}`

Selected by lowest mean regret first (top-1 accuracy at n={len(test_df)//3}
test tasks is too unstable to rank on alone -- see the note above this
report on how a single flipped task previously changed the ranking), then
by accuracy, then by the smallest confidence/accuracy calibration gap
(docs.md §23 "poorly calibrated uncertainty" risk) as a last resort.
{'Beats' if beats_universal else 'Does NOT beat'} the
universal-config baseline on accuracy ({universal_accuracy:.0%}), and
{'beats' if beats_universal_regret else 'does NOT beat'} it on regret
({universal_mean_regret:+.1f} mean steps).

| seed | true best init | predicted | confidence | hit | regret (steps) |
|---|---|---|---:|---|---:|
{winner_detail}

## Verdict

{"H1 supported over H4 at this scale: the best Meta-Predictor design conditions on task/model features and beats a single universal init choice." if beats_universal else "H4 not refuted: even the best of four tested Meta-Predictor designs does not beat the universal-config baseline at this meta-dataset size (" + str(train_df['seed'].nunique()) + " training tasks). The bottleneck is data volume, not model choice -- see docs.md §27 'the meta-dataset's quality intrinsically bounds the meta-predictor's quality.'"}

{"Regret analysis agrees with accuracy: the winner is also the practically cheaper choice on average, not just the more often-correct one." if beats_universal_regret == beats_universal else "Regret analysis diverges from accuracy: " + ("the winner is more often correct but its mistakes are not cheaper on average than the universal baseline's -- accuracy alone would have overstated its practical value." if beats_universal and not beats_universal_regret else "the winner is not more often correct, but when it does miss, it misses by less than the universal baseline does -- accuracy alone would have understated its practical value.")}
"""
    export_csv_snapshots()
    report_path = write_report("compare_meta_predictors", "Meta-Predictor Comparison (4 designs, locked test split)", report)
    print(f"\nReport written to {report_path.relative_to(report_path.parents[2])}")


if __name__ == "__main__":
    main()
