## Recommended stack for PRECOG — with evidence

I structure the stack by architectural layer (cf. docs.md §8-9), deliberately distinguishing the **research environment** (Python) from the **production port** (Rust), as the spec already anticipates (§2: *"your initial goal is not framework performance, but the scientific validation of the hypothesis"*). For each layer, I give the evidence that justifies the choice over an alternative.

---

### 1. Compute substrate — PyTorch (not JAX, not native Rust to start)

**Choice: PyTorch + the `torch.func` ecosystem**

Evidence: `torch.func` (formerly functorch) provides `jacrev`/`jacfwd` composable with `vmap` to compute **batched Jacobians** without a Python loop — exactly what the Trainability Engine (§9.4, §11) needs for gradient/Jacobian/conditioning statistics on a mini-batch:

`jacrev()` can be composed with `vmap` to produce batched Jacobians, and if you run into memory issues computing the Jacobian, you can specify a non-zero `chunk_size`. Per the official PyTorch tutorial documentation, the vectorized version is significantly faster than a manual loop over the Jacobian's rows — the `vmap` version is much faster than the non-vmap version, and becomes even faster as the number of outputs grows.

This is directly the right tool for PRECOG-0 (PURE mode): forward pass, Jacobian, conditioning, without ever calling `optimizer.step()`.

---

### 2. Meta-Predictor & uncertainty — GPyTorch + ensembles

**Choice: GPyTorch for Gaussian processes (small meta-datasets), + network ensembles/quantile regression for large volumes**

GPyTorch is the Gaussian inference engine natively used by BoTorch, so there's no seam to sew between "prediction with uncertainty" and "search engine" — both share the same PyTorch tensor end to end (see point 3).

---

### 3. Search Engine (Bayesian Optimization) — BoTorch + Ax as the core, Optuna as an optional lightweight layer

**Choice: BoTorch (low level) driven by Ax (high level)**

This is exactly the "engine / chassis" architecture the spec describes in §9.8 (Vizier/BO as an exploration arm, not the brain):

BoTorch implements modular building blocks for modern Bayesian optimization. It bridges research and production by being both a highly flexible research framework and a reliable, production-quality implementation. Ax is a sequential experimentation platform that relies on BoTorch to implement Bayesian optimization algorithms, but provides higher-level APIs for specifying problems, visualizing results, and benchmarking new algorithms; it also includes powerful metadata and results-storage management.

The project is actively maintained — the changelog shows releases as recent as June 2026, with Pyro replaced by NumPyro for fully Bayesian NUTS inference, bringing a significant reduction in fit time, and a requirement of PyTorch>=2.4. This matters for PRECOG: the "fully Bayesian" part (SAASBO) is precisely meant to handle the high-dimensional search spaces of §10 (architecture + optimization + regularization combined).

**Optuna** remains relevant as a **lightweight rapid-ablation engine** (§19) when you just want to compare PRECOG variants without standing up the whole Ax infrastructure — but not as the main engine, since Ax natively provides the meta-data management PRECOG needs to build anyway.

---

### 4. Scientific benchmarks — **do not reuse NAS-Bench-201/HPOBench as-is**

This is the most important correction relative to the first version of the document: the two reference benchmarks cited (§15.2) are today **deprecated or nearly inactive**.

- NAS-Bench-201: since NAS-Bench-201 was extended into NATS-Bench, this repository is deprecated and unmaintained; it is recommended to use NATS-Bench, which contains 5x more architecture information and a faster API.
- HPOBench: the latest GitHub releases date back several years; version 0.0.10 fixes handling of a PPO agent parameter, and the commit signing key has expired — a clear sign of a barely-active project.

**Corrected recommendation:**
| Old choice | Recommended replacement | Evidence |
|---|---|---|
| NAS-Bench-201 | **NATS-Bench** | official successor, maintained, 5x more data |
| — | **NAS-Bench-Suite-Zero** (automl/NASLib) | the authors plan to actively maintain the repository and welcome community contributions — it is also the benchmark *specifically designed* to evaluate zero-cost proxies, so directly aligned with PRECOG-0 |
| HPOBench | **HPO-B** or **JAHS-Bench** | JAHS-Bench is cited in recent literature (2025) as an active benchmark for joint architecture+hyperparameter optimization, which matches exactly the multi-level search space of §10 |

A practical shortcut useful for fast prototyping: `simple-hpo-bench`, which provides a set of single-objective HPO benchmark datasets including HPOBench, HPOLib, and NAS-Bench-201 behind a unified API — useful in V1/V2 (§25) before investing in NATS-Bench/JAHS in V4-V5.

---

### 5. Meta-dataset & tracking — reuse what exists rather than reinvent it

Given your MLOps architecture already in place (MLflow → ArgoCD, K8s GPU cluster), the question isn't "which tool" but "should we add a new one at all". Answer: no.

MLflow is recommended if you need to self-host, keep every metric and artifact within your own network, or avoid per-seat billing — exactly your context (data sovereignty over experiment data, already self-hosted). Compared to W&B: MLflow is fully open-source and self-hosted under the Apache 2.0 license, giving full control over the ML infrastructure, whereas W&B offers a managed experience.

**Recommendation:** keep MLflow as the experiment registry (compatible with your existing ArgoCD toolchain), backed by **Postgres** for structured meta-dataset metadata (§12) + **DuckDB/Parquet** for embedded offline analysis (ablations, correlations, ad hoc queries over millions of experiment rows without standing up an analytics cluster).

---

### 6. Orchestration & infra — reuse `ai-helm`, don't build a new one

The PRECOG pipeline (§14) is fundamentally a DAG of experiments with adaptive budgets (FULL TRAINING, PROBE): exactly what your existing architecture (StatefulSet serving, Kubernetes GPU orchestration with MIG/KEDA/Volcano/Kueue) already knows how to do for the dynamic GPU resource-allocation part. The adaptive Short-Probe (§9.10) maps naturally onto Kueue/Volcano for prioritizing short vs. long jobs.

---

### 7. Production port — Rust, consistent with UMC, but **not from V1**

This is where your profile changes the answer relative to a "generic" one. The choice isn't Candle *or* Burn *or* tch-rs — it's a question of phase:

| Phase | Need | Choice | Evidence |
|---|---|---|---|
| Research (V1-V4) | Full autodiff, Jacobian, GP, BO | **PyTorch** | no Rust competitor today has the equivalent of `torch.func`/GPyTorch/BoTorch |
| PURE inference in production (V5-V6, once the meta-predictor is frozen) | Load a trained model, compute zero-cost proxies and score, without training | **tch-rs** if you must reuse trained PyTorch weights as-is, **Candle** if you want HuggingFace-native | for performance, particularly in GPU-accelerated environments, tch-rs is the clearest choice because it relies on PyTorch's highly optimized backend, whereas Candle benefits from strong HuggingFace ecosystem support |
| Long-term native Rust component (aligned with UMC) | Zero Python dependency in production, consistent with your "native Rust only" philosophy | **Burn** | in 2026, Burn reaches v0.15.0 with CPU/GPU/WebAssembly support via CubeCL and a "production readiness" judged high, versus Candle judged "moderate" in production readiness — but by its own documentation Burn still lags on training large models, currently lacking maturity and optimization for large-scale GPU-intensive projects |

**Concrete recommendation:** don't try to go pure-Rust starting at V1 — PRECOG's scientific value (§14-19) must be demonstrated in PyTorch, where the uncertainty/BO ecosystem is incomparably more mature. The Rust port (probably Burn, for consistency with UMC and because its production roadmap is the most favorable of the three per the 2026 comparison) only makes sense at V5-V6, for the **PURE inference** part only (zero-cost scoring + prediction, not training the meta-predictor itself).

---

### 8. Summary

| Layer | Choice | Rejected alternative | Reason |
|---|---|---|---|
| Compute/autodiff | PyTorch + `torch.func` | JAX | more mature and interoperable BO/GP ecosystem (BoTorch/GPyTorch) |
| Uncertainty | GPyTorch (+ ensembles) | — | native to BoTorch |
| Search engine | BoTorch + Ax | open-source Vizier, Optuna alone | Ax natively handles the meta-data management PRECOG needs anyway |
| Benchmarks | NATS-Bench, NAS-Bench-Suite-Zero, JAHS-Bench/HPO-B | NAS-Bench-201, HPOBench | deprecated / poorly maintained |
| Tracking/meta-dataset | MLflow + Postgres + DuckDB | W&B | consistent with your existing infra, self-hosted, no per-seat cost |
| Orchestration | existing K8s (ai-helm) | a new dedicated system | direct reuse of the MIG/Kueue mechanisms already in place |
| Production/inference | Burn (target), tch-rs (transition) | Candle alone | better 2026 "production readiness" trajectory, consistent with UMC's native-Rust philosophy |

The single most immediately actionable point: fix the benchmark choice (§15.2 and §18 of the document) — NATS-Bench + NAS-Bench-Suite-Zero instead of NAS-Bench-201/HPOBench, since the latter will no longer receive updates or active community support.
