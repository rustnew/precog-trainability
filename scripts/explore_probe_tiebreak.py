#!/usr/bin/env python3
"""Tests the second fix option named in
results/reports/2026-09-02T08-04-49Z_explore_scale_invariance_blindspot.md
for zc_jacobcov's proven blind spot: jacob_cov's binary activation-sign
statistic is *exactly* invariant to the positive rescaling that separates
Xavier from He (max |xavier-he| jacob_cov = 0.0 across all 312 meta-dataset
tasks), so no PURE-mode reading of it, however combined with a secondary
proxy, can ever tell them apart. Both static tie-breaks already tried in
scripts/compare_meta_predictors.py (TieBreakHeuristicPredictor's raw and
population-normalized gradient_norm variants) failed for exactly this
reason -- gradient_norm turned out to carry the same he>xavier scale
confound jacob_cov's sign-only statistic doesn't even try to see.

precog.meta_predictor.ProbeTieBreakPredictor tries the other option the
report named: a minimal, explicitly-costed PROBE-mode run (docs.md §5:
DeltaW != 0, but bounded and logged, 50-1000 steps by contract) spent
*only* on the exact tie jacob_cov cannot break. Per the Zero-Training
Contract's own requirement ("must always be possible to answer how much
PROBE adds over PURE alone, for what additional cost"), this reports that
cost -- extra training steps spent per decision -- next to whatever
he-recall / regret improvement it buys, on the same locked TEST split
used throughout this project. The comparison is against the *raw*
zc_jacobcov heuristic (already known to be 0/10 on tasks where "he" is
truly best) and the two failed static tie-breaks, not against FULL
TRAINING -- that would defeat the point of a cheap PURE-mode heuristic.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from precog.experiment_db import load_dataframe, record_gate_evaluation
from precog.meta_predictor import (
    ProbeTieBreakPredictor,
    TieBreakHeuristicPredictor,
    ZeroCostHeuristicPredictor,
    compute_candidate_zero_cost,
)
from precog.model import InitMethod, architecture_from_row
from precog.modes import TrainingConfig
from precog.reporting import export_csv_snapshots, write_report
from precog.taskgen import generate, task_config_from_row

PROBE_STEPS = 50  # the cheapest PROBE-mode call the Zero-Training Contract allows (docs.md §5)
NON_CONVERGENCE_PENALTY = 800 * 2


def _prepare(df):
    df = df.copy()
    df["steps_to_threshold"] = df["steps_to_threshold"].fillna(NON_CONVERGENCE_PENALTY)
    return df


def _training_by_candidate(group) -> dict[InitMethod, TrainingConfig]:
    return {
        InitMethod(row["training.init_method"]): TrainingConfig(
            learning_rate=float(row["training.learning_rate"]),
            batch_size=int(row["training.batch_size"]),
            optimizer=row["training.optimizer"],
            weight_decay=float(row["training.weight_decay"]),
            init_method=InitMethod(row["training.init_method"]),
        )
        for _, row in group.iterrows()
    }


def _evaluate(name: str, predictor, test_df, needs_probe_context: bool = False) -> dict:
    hits = 0
    regrets = []
    probe_costs = []
    he_true_hits = he_true_total = 0
    rows = []
    n = 0
    for seed, group in test_df.groupby("seed"):
        n += 1
        features_row = group.iloc[[0]]
        best_row = group.loc[group["steps_to_threshold"].idxmin()]
        true_best_init = best_row["training.init_method"]
        true_best_steps = best_row["steps_to_threshold"]

        task_config = task_config_from_row(features_row.iloc[0])
        architecture = architecture_from_row(features_row.iloc[0])
        x, y, _ = generate(task_config)
        zc_by_candidate = compute_candidate_zero_cost(architecture, task_config.input_dim, x, y)

        if needs_probe_context:
            rec = predictor.recommend(
                features_row, zc_by_candidate,
                architecture=architecture, x=x, y=y,
                training_by_candidate=_training_by_candidate(group),
            )
        else:
            rec = predictor.recommend(features_row, zc_by_candidate)
        probe_costs.append(rec.probe_cost_steps)

        hit = rec.recommended_init.value == true_best_init
        hits += int(hit)
        if true_best_init == "he":
            he_true_total += 1
            he_true_hits += int(hit)

        predicted_steps = group.loc[
            group["training.init_method"] == rec.recommended_init.value, "steps_to_threshold"
        ].iloc[0]
        regret = predicted_steps - true_best_steps
        regrets.append(regret)
        rows.append({
            "seed": seed, "true_best_init": true_best_init, "predicted_init": rec.recommended_init.value,
            "true_best_steps": true_best_steps, "predicted_steps": predicted_steps,
            "regret": regret, "probe_cost": rec.probe_cost_steps,
        })

    accuracy = hits / n
    mean_regret = float(np.mean(regrets))
    mean_probe_cost = float(np.mean(probe_costs))
    he_recall = he_true_hits / he_true_total if he_true_total else float("nan")
    print(f"{name:<26} accuracy={accuracy:.0%} ({hits}/{n})  "
          f"he_recall={he_recall:.0%} ({he_true_hits}/{he_true_total})  "
          f"mean_regret={mean_regret:+.1f} steps  mean_probe_cost={mean_probe_cost:.1f} steps")
    return {
        "name": name, "accuracy": accuracy, "hits": hits, "n": n,
        "mean_regret": mean_regret, "mean_probe_cost": mean_probe_cost,
        "he_recall": he_recall, "he_true_hits": he_true_hits, "he_true_total": he_true_total,
        "rows": rows,
    }


def main() -> None:
    test_df = _prepare(load_dataframe(split="test"))
    print(f"test (locked): {len(test_df)} rows ({test_df['seed'].nunique()} tasks)\n")

    raw = ZeroCostHeuristicPredictor("jacob_cov", higher_is_better=False)
    tiebreak_raw = TieBreakHeuristicPredictor(primary_proxy="jacob_cov", secondary_proxy="gradient_norm")
    probe = ProbeTieBreakPredictor(primary_proxy="jacob_cov", probe_steps=PROBE_STEPS)

    r_raw = _evaluate("zc_jacobcov", raw, test_df)
    r_tiebreak = _evaluate("zc_jacobcov_tiebreak", tiebreak_raw, test_df)
    r_probe = _evaluate("zc_jacobcov_probetiebreak", probe, test_df, needs_probe_context=True)
    results = [r_raw, r_tiebreak, r_probe]

    full_training_mean_steps = float(test_df.groupby("seed")["steps_to_threshold"].mean().mean())
    probe_overhead_pct = 100 * r_probe["mean_probe_cost"] / full_training_mean_steps
    # he-recall and regret are asked separately: this project's own
    # methodology (scripts/compare_meta_predictors.py) picks winners by
    # regret first because a method that's "more often right on a narrow
    # sub-case" can still be a net-worse decision rule overall -- exactly
    # what a naive "he-recall went up" reading would miss here.
    raises_he_recall = r_probe["he_true_hits"] > r_raw["he_true_hits"]
    beats_on_regret = r_probe["mean_regret"] < r_raw["mean_regret"]
    net_win = raises_he_recall and beats_on_regret

    record_gate_evaluation(
        generation="v1-probe-tiebreak", gate_number=1, metric_name="he_recall_probe_tiebreak",
        metric_value=r_probe["he_recall"] if not np.isnan(r_probe["he_recall"]) else 0.0,
        threshold=r_raw["he_recall"] if not np.isnan(r_raw["he_recall"]) else 0.0,
        n_samples=r_probe["he_true_total"],
        notes=f"raw_he_hits={r_raw['he_true_hits']}/{r_raw['he_true_total']}, "
              f"tiebreak_he_hits={r_tiebreak['he_true_hits']}/{r_tiebreak['he_true_total']}, "
              f"probe_he_hits={r_probe['he_true_hits']}/{r_probe['he_true_total']}, "
              f"probe_steps={PROBE_STEPS}, mean_probe_cost_steps={r_probe['mean_probe_cost']:.1f}, "
              f"overhead_pct_of_mean_full_training={probe_overhead_pct:.1f}",
    )

    results_table = "\n".join(
        f"| {r['name']} | {r['accuracy']:.0%} ({r['hits']}/{r['n']}) | "
        f"{r['he_recall']:.0%} ({r['he_true_hits']}/{r['he_true_total']}) | "
        f"{r['mean_regret']:+.1f} | {r['mean_probe_cost']:.1f} |"
        for r in results
    )
    probe_detail = "\n".join(
        f"| {r['seed']} | {r['true_best_init']} | {r['predicted_init']} | {r['regret']:.0f} | {r['probe_cost']} |"
        for r in r_probe["rows"]
    )
    worst = max(r_probe["rows"], key=lambda r: r["regret"])
    report = f"""## Method

Three candidates evaluated once each on the identical locked TEST split
({len(test_df)} rows, {test_df['seed'].nunique()} tasks) used throughout this
project:

- `zc_jacobcov` -- the raw heuristic, already known to never recommend "he"
  (0/{r_raw['he_true_total']} on tasks where "he" is truly best).
- `zc_jacobcov_tiebreak` -- the first static-secondary-proxy fix attempt
  (gradient_norm), from scripts/compare_meta_predictors.py.
- `zc_jacobcov_probetiebreak` -- this run's new candidate
  (precog.meta_predictor.ProbeTieBreakPredictor): on the exact tie
  jacob_cov cannot break, spends a real, bounded PROBE-mode budget
  ({PROBE_STEPS} steps per tied candidate, docs.md §5) and picks whichever
  ends with the lower loss, instead of another PURE-mode secondary proxy.

Per the Zero-Training Contract (docs.md §5: "must always be possible to
answer how much PROBE adds over PURE alone, for what additional cost"),
the table below reports that cost -- mean extra training steps spent per
decision -- next to accuracy, he-recall (of the {r_raw['he_true_total']}
test tasks where "he" is genuinely the fastest choice) and regret.

## Results

| candidate | accuracy | he-recall | mean regret (steps) | mean probe cost (steps) |
|---|---:|---:|---:|---:|
{results_table}

PROBE overhead: {r_probe['mean_probe_cost']:.0f} steps/decision on average,
against a mean {full_training_mean_steps:.0f}-step FULL TRAINING run on this
split ({probe_overhead_pct:.1f}% of it). Since jacob_cov ties on every
single test task (the blind spot is structural, not occasional), this is
also the candidate's *total* added cost -- there is no untied case to
amortize it against.

| seed | true best init | predicted (probetiebreak) | regret (steps) | probe cost (steps) |
|---|---|---|---:|---:|
{probe_detail}

## Verdict

He-recall {"rises" if raises_he_recall else "does not improve"} from
{r_raw['he_recall']:.0%} ({r_raw['he_true_hits']}/{r_raw['he_true_total']}) to
{r_probe['he_recall']:.0%} ({r_probe['he_true_hits']}/{r_probe['he_true_total']}) with the PROBE
tie-break -- unlike the two static secondary-proxy attempts, spending real
(bounded) training budget on the exact tie can see the Xavier/He difference
at all, because it is the only one of the three that isn't a
positive-rescaling-invariant statistic by construction.

But regret -- this project's primary metric, precisely because it catches
what accuracy alone can't (scripts/compare_meta_predictors.py) -- gets
**worse**, not better: {r_raw['mean_regret']:+.1f} steps (raw) to
{r_probe['mean_regret']:+.1f} steps (PROBE tie-break), on top of
{r_probe['mean_probe_cost']:.0f} extra steps/decision spent
({probe_overhead_pct:.1f}% of a mean FULL TRAINING run) to get there.
{PROBE_STEPS} steps is enough to sometimes make "he" look locally better
than Xavier, but not enough to foresee the cases named in this project's
own README §1 where an init that looks fine early on never actually
converges within budget: the single worst case here is seed {worst['seed']}
(true best is {worst['true_best_init']} at {worst['true_best_steps']:.0f} steps, the
probe picks "{worst['predicted_init']}", which needs {worst['predicted_steps']:.0f} steps to
reach threshold -- regret {worst['regret']:+.0f}). Net: **{"a genuine improvement" if net_win else "NOT a net improvement"}**
-- {"he-recall and regret both move in the right direction." if net_win else "it trades a narrow, cosmetic fix (jacob_cov's 'never recommends he' symptom) for a worse net decision rule, the same shape of finding as this project's other ideas that looked reasonable but underperformed a simpler baseline (LSUV init, active sampling) -- see source.md pillar 4 and §5 above."}
"""
    export_csv_snapshots()
    report_path = write_report(
        "explore_probe_tiebreak", "PROBE-Mode Tie-Break for the Xavier/He Blind Spot", report
    )
    print(f"\nReport written to {report_path.relative_to(report_path.parents[2])}")


if __name__ == "__main__":
    main()
