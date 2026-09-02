# Generalizing the Xavier/He Blind Spot Across Zero-Cost Proxies

_Generated 2026-09-02T08-04-49Z (UTC)_

## Method

Generalizes the root cause found behind zc_jacobcov's inability to ever
recommend "he" (jacob_cov's binary activation-sign statistic is invariant
to rescaling every layer's weights by a positive constant, and Xavier/He
differ from each other by exactly such a rescaling -- same underlying
Gaussian draws, different positive std, zero biases). Checks every other
zero-cost proxy in the Trainability Engine for the same structural
blindness, using the full meta-dataset (312 tasks) already collected
-- no new compute.

## Results

| proxy | max |he-xavier| (raw) | mean relative diff | status |
|---|---:|---:|---|
| jacob_cov | 0 | 0.0000% | BLIND |
| effective_rank | 1.14441e-05 | 0.0000% | BLIND |
| jacobian_condition_mean | 1.16121e+06 | 0.7809% | BLIND |
| activation_mean | 0.277064 | 101.2674% | discriminates |
| gradient_norm | 28.6426 | 125.8556% | discriminates |
| synflow | 253.584 | 140.8736% | discriminates |
| snip | 121.047 | 177.2291% | discriminates |
| activation_variance | 0.573519 | 315.9280% | discriminates |
| gradient_norm_variance | 30010.5 | 429.2232% | discriminates |
| hessian_trace | 293.85 | 432.0215% | discriminates |
| grasp | 14318.7 | 776.6274% | discriminates |

## Verdict

**3/11 proxies are structurally or
near-structurally blind** to the Xavier-vs-He distinction: `jacob_cov`, `effective_rank`, `jacobian_condition_mean`.
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
