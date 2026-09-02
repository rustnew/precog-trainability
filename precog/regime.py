"""Regime Detector (docs.md §9.5): classifies the (model, dataset, hardware)
tuple into a learning regime, producing a prior used to constrain the
predicted hyperparameter distribution:

    (Model, Dataset, Hardware) -> Regime -> Hyperparameter Prior

A simple rule-based classifier for V1 (thresholds are coarse buckets, not
learned) -- matching the spec's own examples ("small model/clean data,
large model/noisy data, low data volume"). Becomes a learned classifier
once enough meta-dataset diversity exists to fit one meaningfully (V4,
docs.md §25)."""
from __future__ import annotations


def _bucket_model_size(n_params: int) -> str:
    if n_params < 2000:
        return "small"
    if n_params < 20_000:
        return "medium"
    return "large"


def _bucket_noise(noise_level: float) -> str:
    if noise_level < 0.1:
        return "clean"
    if noise_level < 0.25:
        return "moderate"
    return "noisy"


def _bucket_volume(n_samples: int) -> str:
    if n_samples < 300:
        return "low"
    if n_samples < 600:
        return "medium"
    return "high"


def detect_regime(model_features: dict, task_features: dict, hardware_features: dict) -> dict:
    model_size = _bucket_model_size(model_features["n_params"])
    noise = _bucket_noise(task_features["noise_level"])
    volume = _bucket_volume(task_features["n_samples"])
    return {
        "regime_model_size": model_size,
        "regime_data_noise": noise,
        "regime_data_volume": volume,
        "regime_device": hardware_features["device_type"],
        "regime_label": f"{model_size}_model__{noise}_data__{volume}_volume",
    }
