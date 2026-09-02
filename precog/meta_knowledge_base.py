"""Meta-Knowledge Base (docs.md §9.6, §13): a structured base of all past
experiments with a task-embedding mechanism to retrieve the historical
experiments closest to a new task, used as a search prior (experience
transfer) -- feeding "Prior Knowledge" into the Meta-Predictor/Search
Engine, per §13's diagram:

    New Task -> Task Encoder -> Task Embedding
             -> [Similar Tasks, Meta-Dataset] -> Prior Knowledge -> Optimization

For V1, the "Task Encoder" is the identity on the already-numeric X_data/
X_model descriptors (§9.1/§9.2), standardized; nearest neighbors are found
by Euclidean distance in that standardized space (sklearn.NearestNeighbors).
A learned embedding (e.g. a small network trained end-to-end) is a natural
V4 upgrade (docs.md §25 "Task Embeddings") once there is enough data to
learn one meaningfully -- with a few dozen tasks, a hand-designed feature
vector is more honest than pretending to have learned a representation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

EMBEDDING_COLUMNS = [
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


def embed_task(features_row: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Task Encoder (§13): the standardized numeric descriptor vector."""
    return scaler.transform(features_row[EMBEDDING_COLUMNS])


class MetaKnowledgeBase:
    """Wraps a set of past experiments (one row per (task, candidate H) pair)
    with a nearest-neighbor index over their task/model embeddings."""

    def __init__(self, k: int = 5):
        self.k = k
        self.scaler = StandardScaler()
        self.index: NearestNeighbors | None = None
        self.experiments: pd.DataFrame | None = None

    def fit(self, experiments: pd.DataFrame) -> None:
        self.experiments = experiments.reset_index(drop=True)
        self.scaler.fit(self.experiments[EMBEDDING_COLUMNS])
        embeddings = self.scaler.transform(self.experiments[EMBEDDING_COLUMNS])
        self.index = NearestNeighbors(n_neighbors=min(self.k, len(embeddings)))
        self.index.fit(embeddings)

    def neighborhood_prior(self, features_row: pd.DataFrame) -> dict:
        """Retrieves the k nearest historical experiments to `features_row`
        (excluding exact self-matches when querying a training row) and
        summarizes them into a prior: the most common best init_method among
        neighbors and their mean best steps_to_threshold -- the "Prior
        Knowledge" handed to the Meta-Predictor (§13)."""
        query = embed_task(features_row, self.scaler)
        distances, indices = self.index.kneighbors(query)
        neighbors = self.experiments.iloc[indices[0]]

        best_per_neighbor_task = neighbors.loc[
            neighbors.groupby("seed")["steps_to_threshold"].idxmin()
        ]
        return {
            "neighborhood_prior_init": best_per_neighbor_task["training.init_method"].mode().iloc[0]
            if not best_per_neighbor_task.empty
            else None,
            "neighborhood_prior_steps_mean": float(best_per_neighbor_task["steps_to_threshold"].mean())
            if not best_per_neighbor_task.empty
            else float("nan"),
            "neighborhood_mean_distance": float(distances.mean()),
        }
