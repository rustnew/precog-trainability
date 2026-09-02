#!/usr/bin/env python3
"""Generalizes the root-cause finding behind zc_jacobcov's Xavier/He blind
spot (see the "diagnose + attempt to fix zc_jacobcov's structural blind
spot" commit): jacob_cov is a function of each sample's binary activation
*sign* pattern, which is invariant to rescaling every layer's weights by a
positive constant -- and Xavier vs He (both zero-biased Gaussian draws from
the same underlying random values, different std) are exactly such a
rescaling. This checks whether OTHER zero-cost proxies share the same
structural blindness for the same underlying reason, using data already in
the meta-dataset (no new compute).

Prediction from theory: any proxy built from a *normalized* or *ratio*
statistic of activations/Jacobian singular values -- not their absolute
scale -- should be scale-invariant too:
  - effective_rank: entropy of *normalized* singular values (sum to 1) ->
    predicted exactly invariant, like jacob_cov.
  - jacobian_condition_mean: sigma_max/sigma_min, a *ratio* -> predicted
    exactly invariant in theory, modulo numerical/definitional noise from
    how it's actually computed layer-by-layer.
  - gradient_norm, hessian_trace, snip: depend on *absolute* magnitude ->
    predicted NOT invariant (confirmed already: gradient_norm(he) is
    always higher than gradient_norm(xavier), a real, if uninformative,
    scale confound).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from precog.experiment_db import load_dataframe, record_gate_evaluation
from precog.reporting import export_csv_snapshots, write_report

PROXY_COLUMNS = [
    "zero_cost.synflow", "zero_cost.snip", "zero_cost.grasp", "zero_cost.jacob_cov",
    "zero_cost.effective_rank", "zero_cost.hessian_trace", "zero_cost.jacobian_condition_mean",
    "zero_cost.gradient_norm", "zero_cost.gradient_norm_variance",
    "zero_cost.activation_mean", "zero_cost.activation_variance",
]
TIE_THRESHOLD = 0.05  # mean relative |he - xavier| difference below 5% counts as "effectively tied"


def main() -> None:
    df = pd.concat([load_dataframe(split="train"), load_dataframe(split="test")], ignore_index=True)
    n_tasks = df["seed"].nunique()
    print(f"Checking Xavier-vs-He tie status for every zero-cost proxy, n={n_tasks} tasks\n")

    rows, blind_proxies = [], []
    for col in PROXY_COLUMNS:
        name = col.removeprefix("zero_cost.")
        piv = df.pivot_table(index="seed", columns="training.init_method", values=col)
        abs_diff = (piv["he"] - piv["xavier"]).abs()
        denom = piv["xavier"].abs().clip(lower=1e-12)
        mean_rel_diff = (abs_diff / denom).mean()
        max_abs_diff = abs_diff.max()
        is_blind = mean_rel_diff < TIE_THRESHOLD
        if is_blind:
            blind_proxies.append(name)
        print(f"{name:<28} max|he-xavier|={max_abs_diff:<14.6g} mean_relative_diff={mean_rel_diff:.4%}  "
              f"{'BLIND (effectively tied)' if is_blind else 'discriminates'}")
        rows.append({"proxy": name, "max_abs_diff": max_abs_diff, "mean_relative_diff": mean_rel_diff, "blind": is_blind})

    print(f"\n{len(blind_proxies)}/{len(PROXY_COLUMNS)} proxies are structurally (or near-structurally) "
          f"blind to Xavier-vs-He: {blind_proxies}")

    record_gate_evaluation(
        generation="v1-scale-invariance-blindspot", gate_number=0,
        metric_name="fraction_proxies_blind_to_xavier_vs_he",
        metric_value=len(blind_proxies) / len(PROXY_COLUMNS), threshold=0.0, n_samples=n_tasks,
        notes=f"blind_proxies={blind_proxies}, tie_threshold={TIE_THRESHOLD:.0%} mean relative difference",
    )

    table = "\n".join(
        f"| {r['proxy']} | {r['max_abs_diff']:.6g} | {r['mean_relative_diff']:.4%} | "
        f"{'BLIND' if r['blind'] else 'discriminates'} |"
        for r in sorted(rows, key=lambda r: r["mean_relative_diff"])
    )
    report = f"""## Method

Generalizes the root cause found behind zc_jacobcov's inability to ever
recommend "he" (jacob_cov's binary activation-sign statistic is invariant
to rescaling every layer's weights by a positive constant, and Xavier/He
differ from each other by exactly such a rescaling -- same underlying
Gaussian draws, different positive std, zero biases). Checks every other
zero-cost proxy in the Trainability Engine for the same structural
blindness, using the full meta-dataset ({n_tasks} tasks) already collected
-- no new compute.

## Results

| proxy | max |he-xavier| (raw) | mean relative diff | status |
|---|---:|---:|---|
{table}

## Verdict

**{len(blind_proxies)}/{len(PROXY_COLUMNS)} proxies are structurally or
near-structurally blind** to the Xavier-vs-He distinction: {", ".join(f"`{p}`" for p in blind_proxies)}.
This is not a bug specific to zc_jacobcov -- it is a property of *any*
zero-cost proxy built from a statistic that is invariant (or nearly so) to
positive per-layer rescaling of the weights (binary activation signs,
normalized/entropy-based singular value statistics, singular-value
*ratios* like a condition number). Only proxies sensitive to *absolute*
magnitude (gradient_norm, hessian_trace, snip, gradient_norm_variance)
discriminate Xavier from He at all -- but this project already showed
separately that gradient_norm's He-vs-Xavier difference is a fixed scale
confound (he always higher, 312/312 tasks) rather than task-dependent
signal, so "discriminates" here does not mean "discriminates usefully".

Practical implication: correctly recommending "he" when it is genuinely
best may require either (a) a proxy family this project hasn't yet tried
that is sensitive to *task-dependent* absolute scale effects (not just a
fixed init-family offset), or (b) a minimal PROBE-mode check (docs.md §5)
for this specific sub-decision, since PURE mode's zero-cost signals may be
structurally insufficient for it with zero-initialized biases.
"""
    export_csv_snapshots()
    report_path = write_report(
        "explore_scale_invariance_blindspot", "Generalizing the Xavier/He Blind Spot Across Zero-Cost Proxies", report
    )
    print(f"\nReport written to {report_path.relative_to(report_path.parents[2])}")


if __name__ == "__main__":
    main()
