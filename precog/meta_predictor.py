"""Meta-Predictor (docs.md §9.7): takes model/task/hardware/regime/zero-cost
features for a *candidate configuration* and predicts its expected outcome,
never as a single value but as a distribution with an attached confidence
(§7.3, §20). Per §9.7's own input list:

    X = [X_model, X_data, X_ZC, X_NEAR, X_init, X_regime]

X_regime comes from the Regime Detector (§9.5); the Meta-Knowledge Base's
neighborhood prior (§9.6, §13's "Prior Knowledge") is added as an extra,
literature-consistent way to inject "similar tasks" experience alongside
the model's own features.

IMPORTANT correction from the first version of this module: X_ZC was
initially left out of the feature set entirely, out of an overcautious
leakage worry carried over from the archived v0 prototype (there,
zero-cost-like features were excluded because the *target* was
init_method itself via classification -- using an init-dependent feature to
classify init would indeed leak). Here the design is different: init_method
is an *input* (one-hot `candidate_init.*`), and the target is that
candidate's own expected steps_to_threshold. The zero-cost score computed
under that exact candidate init is not leakage in this framing -- it is
exactly the signal scripts/gate1_ranking.py validated (§21's controlled
experiment; the individual proxy rho values reported there were later
found to be small-sample overestimates -- see
scripts/gate1_ranking_at_scale.py, which rechecks the same design on the
full 312-task meta-dataset and finds gradient_norm the strongest individual
proxy at rho=0.540, not gradient_norm_variance's originally-reported
0.670). Leaving X_ZC out entirely was still throwing away real, validated
signal even at the corrected magnitude. X_ZC is now included, computed
per-candidate at both fit and predict time.

For V1 scope (§25: Learning Rate, Batch Size, Optimizer, Initialization),
this predicts one head -- T_hat, expected steps_to_threshold -- for a
candidate init_method (optimizer/LR/batch fixed, per §21's controlled
design).

Uncertainty is produced by ensembling (§9.7, §20's first suggested method):
a random forest's per-tree predictions give a natural, cheap ensemble
without hand-rolling bootstrap resampling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from precog.meta_knowledge_base import MetaKnowledgeBase
from precog.model import InitMethod
from precog.modes import Mode, TrainingConfig, TrainProtocol, train
from precog.regime import _bucket_noise, _bucket_volume
from precog.trainability import zero_cost_features

BASE_FEATURE_COLUMNS = [
    "task.input_dim",
    "task.noise_level",
    "task.n_samples",
    "task.target_variance",
    "task.target_entropy_estimate",
    "task.feature_correlation_mean",
    "task.redundancy",
    "model.depth",
    "model.width",
    "model.n_params",
    "model.flops",
]
# The proxies scripts/gate1_ranking.py's controlled experiment (§21) actually
# validated against real convergence speed (individually significant,
# p < 0.05) -- see results/reports/*_gate1_ranking.md for the numbers this
# selection is based on, not assumed.
ZERO_COST_COLUMNS = [
    "zero_cost.gradient_norm",
    "zero_cost.gradient_norm_variance",
    "zero_cost.jacob_cov",
    "zero_cost.effective_rank",
    "zero_cost.jacobian_condition_mean",
]
_NOISE_BUCKETS = ["clean", "moderate", "noisy"]
_VOLUME_BUCKETS = ["low", "medium", "high"]
REGIME_COLUMNS = [f"regime_noise.{b}" for b in _NOISE_BUCKETS] + [f"regime_volume.{b}" for b in _VOLUME_BUCKETS]
PRIOR_COLUMNS = ["neighborhood_prior_steps_mean"]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ZERO_COST_COLUMNS + REGIME_COLUMNS + PRIOR_COLUMNS
# For the "reduced features" ablation (scripts/compare_meta_predictors.py):
# only the individually-validated zero-cost signals, dropping the
# regime/prior/architecture features that Gate 1 never tested directly.
REDUCED_FEATURE_COLUMNS = ZERO_COST_COLUMNS

CANDIDATE_INIT_METHODS = [InitMethod.XAVIER, InitMethod.HE, InitMethod.ORTHOGONAL]


def engineer_features(features_row: pd.DataFrame, mkb: MetaKnowledgeBase) -> pd.DataFrame:
    """Adds X_regime (§9.5, one-hot -- recomputed from raw task stats rather
    than trusting a stored regime label, so this also works for a brand new
    task never logged in the meta-dataset) and the Meta-Knowledge Base's
    neighborhood prior (§9.6, §13) to the base task/model features.

    X_ZC is handled separately: train time already has it stored per-row
    (computed under the init that row actually used); predict time computes
    it fresh per candidate via compute_candidate_zero_cost()."""
    row = features_row.copy()
    noise_bucket = _bucket_noise(row["task.noise_level"].iloc[0])
    volume_bucket = _bucket_volume(row["task.n_samples"].iloc[0])
    for b in _NOISE_BUCKETS:
        row[f"regime_noise.{b}"] = float(b == noise_bucket)
    for b in _VOLUME_BUCKETS:
        row[f"regime_volume.{b}"] = float(b == volume_bucket)

    prior = mkb.neighborhood_prior(features_row)
    row["neighborhood_prior_steps_mean"] = prior["neighborhood_prior_steps_mean"]
    return row


def compute_candidate_zero_cost(
    architecture, input_dim: int, x: torch.Tensor, y: torch.Tensor
) -> dict[InitMethod, dict]:
    """PURE-mode X_ZC for every candidate init (§9.4/§5, DeltaW=0), used at
    predict time when no stored experiment exists yet for this task."""
    from precog.model import build_mlp

    result = {}
    for candidate in CANDIDATE_INIT_METHODS:
        torch.manual_seed(0)
        model = build_mlp(architecture, candidate)
        result[candidate] = zero_cost_features(model, input_dim, x, y)
    return result


def _with_candidate(features_row: pd.DataFrame, init_method: InitMethod, zc: dict) -> pd.DataFrame:
    row = features_row.copy()
    for candidate in CANDIDATE_INIT_METHODS:
        row[f"candidate_init.{candidate.value}"] = float(candidate == init_method)
    for col in ZERO_COST_COLUMNS:
        row[col] = zc[col.removeprefix("zero_cost.")]
    return row


def _candidate_columns() -> list[str]:
    return [f"candidate_init.{c.value}" for c in CANDIDATE_INIT_METHODS]


@dataclass
class Recommendation:
    """docs.md §9.7's output format: never a point value alone."""

    recommended_init: InitMethod
    expected_steps: float
    steps_range: tuple[float, float]  # +/- 1 std across the ensemble
    confidence: float  # 1 - (relative spread), clamped to [0, 1]
    per_candidate: dict[str, dict]  # every candidate's own prediction, for transparency
    probe_cost_steps: int = 0  # PROBE-mode budget actually spent on this decision (docs.md §5 cost-accounting)


class MetaPredictor:
    """One head for now (T_hat = expected steps_to_threshold); additional
    heads (A_hat, C_hat, N_hat, docs.md §9.7) are a straightforward
    extension of the same ensemble once their ground truth is logged.

    `feature_columns` and `log_target` exist to support the ablation in
    scripts/compare_meta_predictors.py (docs.md §19 "no proxy/feature is
    assumed to help -- test it"): a log-space target (the 1600-step
    non-convergence penalty is a heavy-tailed outlier in raw space) and a
    reduced feature set are both tested as alternatives, not assumed to be
    better than the full feature set."""

    def __init__(
        self,
        n_estimators: int = 300,
        random_state: int = 0,
        feature_columns: list[str] | None = None,
        log_target: bool = False,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, min_samples_leaf=2
        )
        self.feature_columns = feature_columns if feature_columns is not None else FEATURE_COLUMNS
        self.log_target = log_target
        self._fitted_columns: list[str] | None = None

    def fit(self, features: pd.DataFrame, init_methods: pd.Series, steps_to_threshold: pd.Series) -> None:
        """`features` rows must already carry their own zero_cost.* columns
        (true at fit time: they come straight from the logged experiment,
        computed under the init_method that row actually used)."""
        rows = []
        for (_, feat_row), init_value in zip(features.iterrows(), init_methods):
            zc = {col.removeprefix("zero_cost."): feat_row[col] for col in ZERO_COST_COLUMNS}
            row = _with_candidate(feat_row.to_frame().T, InitMethod(init_value), zc)
            rows.append(row)
        x_train = pd.concat(rows, ignore_index=True)[self.feature_columns + _candidate_columns()]
        self._fitted_columns = list(x_train.columns)
        target = np.log1p(steps_to_threshold.to_numpy()) if self.log_target else steps_to_threshold.to_numpy()
        self.model.fit(x_train, target)

    def _predict_with_uncertainty(self, x_row: pd.DataFrame) -> tuple[float, float]:
        x_row = x_row[self._fitted_columns].to_numpy()
        tree_predictions = np.array([tree.predict(x_row)[0] for tree in self.model.estimators_])
        if self.log_target:
            tree_predictions = np.expm1(tree_predictions)
        return float(tree_predictions.mean()), float(tree_predictions.std())

    def recommend(
        self, features_row: pd.DataFrame, zero_cost_by_candidate: dict[InitMethod, dict]
    ) -> Recommendation:
        """docs.md §9.7: 'for each candidate configuration, a multi-head
        prediction' -- query every candidate init (each with its own
        freshly-computed PURE-mode X_ZC, see compute_candidate_zero_cost()),
        keep the best, but report all of them (the Rank/Optimize step, §18
        diagram, needs the full set, not just the winner)."""
        per_candidate = {}
        for candidate in CANDIDATE_INIT_METHODS:
            x_row = _with_candidate(features_row, candidate, zero_cost_by_candidate[candidate])
            mean_steps, std_steps = self._predict_with_uncertainty(x_row)
            per_candidate[candidate.value] = {"expected_steps": mean_steps, "std_steps": std_steps}

        best_init_name = min(per_candidate, key=lambda k: per_candidate[k]["expected_steps"])
        best = per_candidate[best_init_name]
        relative_spread = best["std_steps"] / max(best["expected_steps"], 1e-6)
        confidence = float(np.clip(1.0 - relative_spread, 0.0, 1.0))

        return Recommendation(
            recommended_init=InitMethod(best_init_name),
            expected_steps=best["expected_steps"],
            steps_range=(
                max(0.0, best["expected_steps"] - best["std_steps"]),
                best["expected_steps"] + best["std_steps"],
            ),
            confidence=confidence,
            per_candidate=per_candidate,
        )


class GPMetaPredictor:
    """Gaussian Process regression on steps_to_threshold -- stack.md names
    GPyTorch/BoTorch/Ax as PRECOG's *target* framework (Optuna as a
    "lightweight interim"), a decision never actually tested empirically
    until now. sklearn's exact GP is fast enough at this meta-dataset's
    scale (~250 training tasks) to test the hypothesis without pulling in
    the heavier dependency: does a GP's posterior std -- a principled
    predictive uncertainty, unlike RandomForestRegressor's ad-hoc
    inter-tree spread -- fix this project's recurring calibration problem
    (mean confidence tracking accuracy poorly across every RF variant
    tested in compare_meta_predictors.py so far)?

    log_target defaults to True here (unlike MetaPredictor): a GP's
    Matern kernel assumes roughly homoscedastic, smooth noise, which the
    1600-step non-convergence penalty's heavy right tail badly violates in
    raw space.
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        log_target: bool = True,
        random_state: int = 0,
    ):
        self.feature_columns = feature_columns if feature_columns is not None else REDUCED_FEATURE_COLUMNS
        self.log_target = log_target
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
            noise_level=1.0, noise_level_bounds=(1e-3, 1e2)
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=random_state, n_restarts_optimizer=3
        )
        self.scaler = StandardScaler()
        self._fitted_columns: list[str] | None = None

    def fit(self, features: pd.DataFrame, init_methods: pd.Series, steps_to_threshold: pd.Series) -> None:
        rows = []
        for (_, feat_row), init_value in zip(features.iterrows(), init_methods):
            zc = {col.removeprefix("zero_cost."): feat_row[col] for col in ZERO_COST_COLUMNS}
            row = _with_candidate(feat_row.to_frame().T, InitMethod(init_value), zc)
            rows.append(row)
        x_train = pd.concat(rows, ignore_index=True)[self.feature_columns + _candidate_columns()]
        self._fitted_columns = list(x_train.columns)
        x_scaled = self.scaler.fit_transform(x_train.to_numpy())
        target = np.log1p(steps_to_threshold.to_numpy()) if self.log_target else steps_to_threshold.to_numpy()
        self.model.fit(x_scaled, target)

    def _predict_with_uncertainty(self, x_row: pd.DataFrame) -> tuple[float, float]:
        x = self.scaler.transform(x_row[self._fitted_columns].to_numpy())
        mean, std = self.model.predict(x, return_std=True)
        mean, std = float(mean[0]), float(std[0])
        if self.log_target:
            # First-order (delta-method) transform of the log-space
            # posterior back to raw steps: d(expm1)/dz = exp(z), evaluated
            # at the posterior mean.
            mean_steps = float(np.expm1(mean))
            std_steps = float(mean_steps * std)
        else:
            mean_steps, std_steps = mean, std
        return mean_steps, std_steps

    def recommend(
        self, features_row: pd.DataFrame, zero_cost_by_candidate: dict[InitMethod, dict]
    ) -> Recommendation:
        per_candidate = {}
        for candidate in CANDIDATE_INIT_METHODS:
            x_row = _with_candidate(features_row, candidate, zero_cost_by_candidate[candidate])
            mean_steps, std_steps = self._predict_with_uncertainty(x_row)
            per_candidate[candidate.value] = {"expected_steps": mean_steps, "std_steps": std_steps}

        best_init_name = min(per_candidate, key=lambda k: per_candidate[k]["expected_steps"])
        best = per_candidate[best_init_name]
        relative_spread = best["std_steps"] / max(best["expected_steps"], 1e-6)
        confidence = float(np.clip(1.0 - relative_spread, 0.0, 1.0))

        return Recommendation(
            recommended_init=InitMethod(best_init_name),
            expected_steps=best["expected_steps"],
            steps_range=(
                max(0.0, best["expected_steps"] - best["std_steps"]),
                best["expected_steps"] + best["std_steps"],
            ),
            confidence=confidence,
            per_candidate=per_candidate,
        )


class ZeroCostHeuristicPredictor:
    """No learning at all -- ranks candidates directly by a single raw
    zero-cost proxy, exactly the methodology the zero-cost NAS literature
    itself uses (source.md pillar 3: SynFlow/SNIP/NASWOT papers rank
    architectures by the proxy score directly, no meta-model on top). Worth
    testing on its own merits: the learned RandomForest wrapper (44-48%
    accuracy) barely beats the universal baseline, so it's a fair question
    whether the wrapper is adding value over the raw signal Gate 1 already
    validated (gradient_norm alone: rho=0.540 at full meta-dataset scale,
    see scripts/gate1_ranking_at_scale.py -- the single strongest individual
    proxy found, correcting gate1_ranking.py's original small-sample
    gradient_norm_variance rho=0.670 estimate).

    Needs zero training data at all -- this is the "PRECOG-0" tier
    (docs.md's own §19.1 ablation ladder starts at "Zero-Cost only")."""

    def __init__(self, proxy_name: str = "gradient_norm_variance", higher_is_better: bool = False):
        self.proxy_name = proxy_name
        self.higher_is_better = higher_is_better

    def recommend(self, features_row: pd.DataFrame, zero_cost_by_candidate: dict[InitMethod, dict]) -> Recommendation:
        per_candidate = {
            c.value: {"expected_steps": float("nan"), "std_steps": 0.0, "raw_score": zc[self.proxy_name]}
            for c, zc in zero_cost_by_candidate.items()
        }
        key_fn = lambda k: per_candidate[k]["raw_score"] * (-1 if self.higher_is_better else 1)
        best_init_name = min(per_candidate, key=key_fn)
        return Recommendation(
            recommended_init=InitMethod(best_init_name),
            expected_steps=float("nan"),
            steps_range=(float("nan"), float("nan")),
            confidence=float("nan"),  # this method makes no probabilistic claim -- see docs.md §20
            per_candidate=per_candidate,
        )


class TieBreakHeuristicPredictor:
    """Fixes a root-caused blind spot found in zc_jacobcov (the project's
    best-evidenced method so far): jacob_cov is a function of each sample's
    binary activation *sign* pattern only, which is invariant to rescaling
    every layer's weights by a positive constant. Xavier and He both draw
    from the same underlying Gaussian random values (same shape, same
    global RNG state, same `.normal_()` call under the hood) and only
    differ in the positive std they scale by, with zero-initialized biases
    -- so their jacob_cov score is *exactly* identical, verified across all
    312 meta-dataset tasks (max |xavier - he| jacob_cov = 0.0). This isn't
    noise; jacob_cov structurally cannot ever distinguish them. Since
    InitMethod.XAVIER is listed before InitMethod.HE in
    CANDIDATE_INIT_METHODS, `min()` silently always breaks that exact tie
    in Xavier's favor, which is why the raw zc_jacobcov heuristic never
    once recommends "he" across the entire locked test split, even on the
    10 tasks where "he" is genuinely the true best init.

    `secondary_proxy` breaks ties on a *scale-sensitive* signal instead:
    gradient_norm differs sharply by init (he substantially higher on
    average than xavier/orthogonal in this meta-dataset), so it can
    actually discriminate the exact case jacob_cov cannot.
    """

    def __init__(
        self,
        primary_proxy: str = "jacob_cov",
        secondary_proxy: str = "gradient_norm",
        primary_higher_is_better: bool = False,
        secondary_higher_is_better: bool = False,
        tie_tolerance: float = 1e-6,
        secondary_population_stats: dict[str, tuple[float, float]] | None = None,
    ):
        self.primary_proxy = primary_proxy
        self.secondary_proxy = secondary_proxy
        self.primary_higher_is_better = primary_higher_is_better
        self.secondary_higher_is_better = secondary_higher_is_better
        self.tie_tolerance = tie_tolerance
        # When set, the secondary proxy is z-scored against each candidate's
        # own init-family population stats before breaking ties -- fixes
        # gradient_norm's own fixed scale confound (he > xavier on 312/312
        # tasks by a constant-ish ratio), not just jacob_cov's exact tie.
        self.secondary_population_stats = secondary_population_stats

    def recommend(self, features_row: pd.DataFrame, zero_cost_by_candidate: dict[InitMethod, dict]) -> Recommendation:
        per_candidate = {
            c.value: {
                "expected_steps": float("nan"), "std_steps": 0.0,
                "primary_score": zc[self.primary_proxy], "secondary_score": zc[self.secondary_proxy],
            }
            for c, zc in zero_cost_by_candidate.items()
        }
        primary_sign = -1 if self.primary_higher_is_better else 1
        primary_values = {k: v["primary_score"] * primary_sign for k, v in per_candidate.items()}
        best_primary = min(primary_values.values())
        tied = [k for k, v in primary_values.items() if abs(v - best_primary) <= self.tie_tolerance]

        if len(tied) == 1:
            best_init_name = tied[0]
        else:
            secondary_sign = -1 if self.secondary_higher_is_better else 1

            def secondary_key(k: str) -> float:
                score = per_candidate[k]["secondary_score"]
                if self.secondary_population_stats is not None:
                    mean, std = self.secondary_population_stats[k]
                    score = (score - mean) / (std + 1e-12)
                return score * secondary_sign

            best_init_name = min(tied, key=secondary_key)

        return Recommendation(
            recommended_init=InitMethod(best_init_name),
            expected_steps=float("nan"),
            steps_range=(float("nan"), float("nan")),
            confidence=float("nan"),  # this method makes no probabilistic claim -- see docs.md §20
            per_candidate=per_candidate,
        )


class ProbeTieBreakPredictor:
    """Third fix attempt for zc_jacobcov's proven blind spot (see
    TieBreakHeuristicPredictor above): jacob_cov's binary activation-sign
    statistic is *exactly* invariant to the positive rescaling that
    separates Xavier from He (max |xavier-he| jacob_cov = 0.0 across all
    312 meta-dataset tasks), so no PURE-mode secondary proxy -- raw or
    population-normalized gradient_norm, both tried in
    scripts/compare_meta_predictors.py -- can ever break that exact tie;
    gradient_norm turned out to carry the same he>xavier scale confound
    jacob_cov's sign-only statistic doesn't even look at.

    This tries the other option named in
    results/reports/2026-09-02T08-04-49Z_explore_scale_invariance_blindspot.md:
    a minimal PROBE-mode check (docs.md §5: DeltaW != 0, but bounded and
    logged, 50-1000 steps by contract) spent *only* on the exact tie
    jacob_cov cannot see -- train each tied candidate for `probe_steps`
    real steps at its own (learning_rate, batch_size, optimizer) and keep
    whichever ends with the lower loss. `last_probe_cost_steps` records
    the budget actually spent on the most recent call, so callers can
    report it per the Zero-Training Contract's own requirement ("must
    always be possible to answer how much PROBE adds over PURE alone, for
    what additional cost") -- see scripts/explore_probe_tiebreak.py."""

    def __init__(
        self,
        primary_proxy: str = "jacob_cov",
        primary_higher_is_better: bool = False,
        tie_tolerance: float = 1e-6,
        probe_steps: int = 50,
    ):
        self.primary_proxy = primary_proxy
        self.primary_higher_is_better = primary_higher_is_better
        self.tie_tolerance = tie_tolerance
        self.probe_steps = probe_steps
        self.last_probe_cost_steps = 0

    def recommend(
        self,
        features_row: pd.DataFrame,
        zero_cost_by_candidate: dict[InitMethod, dict],
        architecture=None,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        training_by_candidate: dict[InitMethod, TrainingConfig] | None = None,
    ) -> Recommendation:
        per_candidate = {
            c.value: {"expected_steps": float("nan"), "std_steps": 0.0, "primary_score": zc[self.primary_proxy]}
            for c, zc in zero_cost_by_candidate.items()
        }
        primary_sign = -1 if self.primary_higher_is_better else 1
        primary_values = {k: v["primary_score"] * primary_sign for k, v in per_candidate.items()}
        best_primary = min(primary_values.values())
        tied = [k for k, v in primary_values.items() if abs(v - best_primary) <= self.tie_tolerance]

        self.last_probe_cost_steps = 0
        if len(tied) == 1:
            best_init_name = tied[0]
        else:
            if architecture is None or x is None or y is None or training_by_candidate is None:
                raise ValueError(
                    "ProbeTieBreakPredictor needs a live architecture/x/y/training_by_candidate "
                    "context to actually run the PROBE that breaks the tie -- pass them through, "
                    "see scripts/explore_probe_tiebreak.py for how the harness wires this up."
                )
            probe_losses = {}
            for k in tied:
                training = training_by_candidate[InitMethod(k)]
                protocol = TrainProtocol(
                    mode=Mode.PROBE, max_steps=self.probe_steps, loss_threshold=-1.0, seed=0
                )
                result = train(architecture, x, y, training, protocol)
                probe_losses[k] = result.final_loss
                self.last_probe_cost_steps += self.probe_steps
            best_init_name = min(probe_losses, key=probe_losses.get)

        return Recommendation(
            recommended_init=InitMethod(best_init_name),
            expected_steps=float("nan"),
            steps_range=(float("nan"), float("nan")),
            confidence=float("nan"),  # this method makes no probabilistic claim -- see docs.md §20
            per_candidate=per_candidate,
            probe_cost_steps=self.last_probe_cost_steps,
        )


class KNNMetaPredictor:
    """Alternative to the RandomForest MetaPredictor (docs.md §19 ablation
    spirit): predicts purely from the Meta-Knowledge Base's (§9.6) nearest
    neighbors, with no learned model of its own -- the simplest possible
    baseline that still uses task similarity, worth testing given how few
    training tasks exist so far (a 300-tree forest over 18 features on ~70
    tasks is a lot of capacity for the data available)."""

    def __init__(self, mkb: MetaKnowledgeBase):
        self.mkb = mkb

    def recommend(self, features_row: pd.DataFrame, zero_cost_by_candidate: dict[InitMethod, dict]) -> Recommendation:
        prior = self.mkb.neighborhood_prior(features_row)
        recommended = InitMethod(prior["neighborhood_prior_init"]) if prior["neighborhood_prior_init"] else InitMethod.XAVIER
        expected_steps = prior["neighborhood_prior_steps_mean"]
        return Recommendation(
            recommended_init=recommended,
            expected_steps=expected_steps,
            steps_range=(expected_steps, expected_steps),
            confidence=1.0 / (1.0 + prior["neighborhood_mean_distance"]),
            per_candidate={recommended.value: {"expected_steps": expected_steps, "std_steps": 0.0}},
        )
