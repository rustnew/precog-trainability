# Reference Bibliography

No single document describes PRECOG exactly: it sits at the intersection of several research areas. This is the reference corpus surveyed while writing `docs.md`, organized by the eight areas that inform PRECOG's design. It complements `docs.md` (specification) and `stack.md` (technology choices) — it does not repeat their content.

Primary sources gathered for this review (arXiv / PMLR):

```
https://arxiv.org/abs/2107.05847
https://arxiv.org/html/2410.22854v1
https://proceedings.mlr.press/v70/wichrowska17a.html
https://proceedings.mlr.press/v139/sandler21a.html
https://arxiv.org/abs/2101.08134
https://arxiv.org/html/2307.01998v3
https://arxiv.org/pdf/2110.10423
https://arxiv.org/abs/2404.00271
https://arxiv.org/pdf/1711.04735
https://arxiv.org/html/2508.02882v2
```

---

## 1. Hyperparameter Optimization

The field closest to Google Vizier.

- **Google Vizier: A Service for Black-Box Optimization** (Golovin et al., Google Research) — *foundational*. Describes Google's engine for automatically optimizing hyperparameters and complex objective functions via black-box optimization. The single most important industrial baseline for PRECOG (see `docs.md` §6, §18).
- **Hyperparameter Optimization: Foundations, Algorithms, Best Practices and Open Challenges** (Bischl et al., arXiv:2107.05847) — a comprehensive survey covering grid search, random search, Bayesian optimization, Hyperband, and open challenges. A theoretical reference, not a new algorithm; useful bibliography for `docs.md` §6/§18, not directly actionable code.
- **Hyperparameter Optimization in Machine Learning** (Franceschi et al., arXiv:2410.22854) — a recent monograph formalizing HPO as minimizing $f(\lambda) = M(A(D,\lambda), V)$, covering grid/random search, Bayesian optimization (GP), multi-fidelity methods (Hyperband/ASHA), CMA-ES, and hypergradients (differentiating through the training loop to optimize hyperparameters directly by gradient descent). No unified formula — a conceptual framework, not a single algorithm. The **hypergradients** angle is a research direction distinct from PRECOG's meta-learning approach, worth keeping as a reference (see `docs.md` §36-equivalent open questions).

## 2. Meta-Learning: Learning from Past Experience

PRECOG's philosophy leans heavily on one idea: *past experience must become reusable knowledge.*

- **Initializing Bayesian Hyperparameter Optimization via Meta-Learning** (AAAI) — shows that past tasks can be used to warm-start optimization in promising regions instead of starting from scratch. Exactly the logic behind PRECOG's Meta-Knowledge Base (`docs.md` §9.6, §13).
- **Learned Optimizers that Scale and Generalize** (Wichrowska et al., ICML 2017, PMLR v70) — proposes an optimizer itself learned by a neural network, meant to generalize to new tasks. Relevant if PRECOG evolves toward a learned optimizer rather than a hyperparameter predictor — an ambitious, explicitly out-of-scope-for-now direction (`docs.md` §3.3).
- **Meta-Learning Bidirectional Update Rules** (Sandler et al., ICML 2021, PMLR v139) — studies learned update rules in place of classic gradient-descent rules. Relevant for a future PRECOG version where the update itself becomes predictive.

## 3. Training-Free Prediction

Likely the branch closest to PRECOG's original idea: predicting a network's performance without fully training it.

- **Zero-Cost Proxies for Lightweight NAS** (Abdelfattah, Mehrotra, Dudziak, Lane, ICLR 2021, arXiv:2101.08134) — *very important, directly actionable*. Introduces scores computed from a single minibatch (SynFlow, SNIP, GraSP, Jacob-Cov, grad-norm, Fisher) that rank architectures without any training. Their best proxy reaches a Spearman correlation of **0.82** with final accuracy on NAS-Bench-201 (vs. 0.61 for EcoNAS, a reduced-training method), matching EcoNAS's accuracy **4x faster** on NAS-Bench-101. This is the direct basis for `precog/trainability.py`'s SynFlow/SNIP/GraSP implementations.
- **Zero-Shot Neural Architecture Search** — a comprehensive survey of methods for predicting architecture quality without training parameters; a scientific map of training-free proxies.
- **NEAR: A Training-Free Pre-Estimator of Machine Learning Model Performance** — one of the most interesting papers for PRECOG. Proposes a score based on the effective rank of activations to estimate a network's future performance, also demonstrated for selecting initialization and activation functions (`docs.md` §9.4 "NEAR").
- **ProxyBO** — combines zero-cost proxies with Bayesian Optimization to accelerate architecture search. Close to the idea of using analytical signals before investing in costly training.
- **TG-NAS** — uses a Transformer and a GCN as a universal proxy to predict architecture performance without retraining the predictor for each search space. Relevant to PRECOG's Model Encoder (`docs.md` §9.1).

## 4. Initialization and Dynamical Isometry

Why do some initializations enable much faster convergence?

- **Provable Benefit of Orthogonal Initialization in Optimizing Deep Linear Networks** — *foundational*. Mathematically demonstrates that orthogonal initialization can accelerate convergence in deep linear networks compared to certain Gaussian initializations. A key reference for `precog/model.py`'s `InitMethod.ORTHOGONAL`.
- **Resurrecting the Sigmoid in Deep Learning Through Dynamical Isometry** (Pennington, Schoenholz, Ganguli) — studies signal propagation and shows that certain initialization conditions enable better gradient flow and much faster learning in some regimes. The theoretical basis of dynamical isometry (`docs.md` §6, §11.2).
- **On the Neural Tangent Kernel of Deep Networks with Orthogonal Initialization** — analyzes the link between orthogonality, the NTK, and learning speed. Useful if PRECOG wants to exploit geometric properties before training.
- **Dynamical Isometry and a Mean Field Theory of RNNs** — develops a theory of signal propagation at initialization using random matrix theory and mean-field methods. Important for understanding how to analyze an untrained network.

## 5. Training Dynamics and Learning Geometry

PRECOG wants to measure what happens in the first steps of learning to predict what follows. Associated fields: Gradient Flow, Loss Landscape, Hessian Spectrum, Jacobian Analysis, Sharpness, Curvature, Gradient Noise, Gradient Alignment.

- **Deep Network Trainability via Persistent Subspace Orthogonality** — recent work deepening the notion of dynamical isometry and the relationship between persistent orthogonality and trainability. Shows that a network's internal geometry remains an active research topic.

## 6. Sample Efficiency

The goal isn't only reducing epochs — it's also reducing $N_\epsilon$, the amount of data needed to reach a target performance.

- **A Survey of Deep Active Learning** — active learning explicitly seeks to maximize the performance gain per labeled example. Essential literature for PRECOG's sample-efficiency axis (`docs.md` §16 "Data efficiency").
- **A Comparative Survey of Deep Active Learning** — compares many active-learning methods under a homogeneous experimental setup. Useful for building PRECOG's future benchmarks.

## 7. Training-Free Neural Architecture Search

Even though PRECOG doesn't primarily search for the best architecture, this literature supplies the methods for predicting an untrained network's quality.

- **NASWOT and Zero-Cost Proxies** — introduces scores computed at initialization to rank architectures without full training. A direct methodological inspiration for PRECOG.
- **Generic Neural Architecture Search via Regression** — explores predicting architecture performance via regression, using network representations rather than full training runs.
- **RBFleX-NAS** — recent work on training-free approaches for selecting architectures at minimal cost.

## 8. Open-Source Tools to Study

These projects don't share all of PRECOG's philosophy, but they are the reference infrastructure.

- **Google Vizier (OSS)** — open-source black-box optimization framework inspired by Google's internal system. Study for: hyperparameter search engine, experimentation API.
- **Optuna** — widely used framework for reproducible experimental studies and modern tuning algorithms. Used as an experimental baseline (see `stack.md` §3).
- **Ray Tune** — distributed library for running thousands of parallel experiments across GPUs or clusters.
- **Kubeflow Katib** — Kubernetes-native tuning system for MLOps pipelines. Relevant for enterprise industrialization (see `stack.md` §6).
- **Awesome AutoML** — an open-source collection of the main AutoML, HPO, NAS, and benchmark frameworks. A useful bibliographic index.

---

## The common philosophy

Every one of these lines of work can be summarized by the same idea: *analyze before spending compute.* The essential difference between them:

| Field | Question asked |
|---|---|
| Google Vizier | Which hyperparameters give the best result after trials? |
| Optuna / Ray Tune | How do we search efficiently for the best configurations? |
| Zero-Cost NAS | Can a network be evaluated without training it? |
| Dynamical Isometry | Which initialization favors gradient propagation? |
| Learned Optimizers | Can we learn a better optimization rule? |
| Active Learning | How do we reach a target performance with less data? |
| **PRECOG** | **Can we predict, before training, the best convergence conditions by combining all of this knowledge?** |

PRECOG does not invent a new field; it seeks to unify several currently separate branches of research into a single system that predicts training dynamics.
