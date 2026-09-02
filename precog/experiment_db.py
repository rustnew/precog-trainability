"""The meta-dataset: PRECOG's scientific memory (docs.md §12).

Every experiment -- including every failure -- is recorded. Backed by SQLite
here (a local stand-in for the Postgres + MLflow combination stack.md §5
recommends for the production cluster this sandbox doesn't have access to);
the schema is what matters, not the specific database engine.

Strict separation (docs.md §12): rows are tagged with a `split` column
(train/validation/test) at insertion time, and the test split must never be
queried while developing or tuning the meta-predictor -- only for final,
one-shot evaluation (§15.1, §26).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "meta_dataset.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    seed INTEGER NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('probe', 'full_training')),

    -- Model (docs.md §9.1)
    model_json TEXT NOT NULL,

    -- Dataset / task (docs.md §9.2)
    task_json TEXT NOT NULL,

    -- Hardware (docs.md §9.3)
    hardware_json TEXT NOT NULL,

    -- Regime (docs.md §9.5)
    regime_json TEXT NOT NULL,

    -- Hyperparameters under test (docs.md §10.1)
    training_json TEXT NOT NULL,

    -- Zero-cost descriptors, PURE mode (docs.md §9.4)
    zero_cost_json TEXT,

    -- Outcome / ground truth (docs.md §12 "Ground truth")
    initial_loss REAL,
    final_loss REAL,
    steps_to_threshold INTEGER,
    converged INTEGER NOT NULL,
    diverged INTEGER NOT NULL,
    wall_clock_s REAL,
    delta_w_norm REAL
);

-- Progression gates (docs.md §17): "each gate is validated by independent
-- metrics, on locked datasets, before considering the next generation" --
-- so the history of every check must persist, not just the latest run's
-- console output, to actually compare generations over time.
CREATE TABLE IF NOT EXISTS gate_evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    generation TEXT NOT NULL,
    gate_number INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    threshold REAL NOT NULL,
    passed INTEGER NOT NULL,
    n_samples INTEGER,
    notes TEXT
);

-- Search Engine trials (docs.md §9.8, §12 "every experiment must be
-- recorded"): kept in a *separate* table from `experiments`, not tagged
-- with a split value, because these are candidate configs a search
-- explored on a task -- not the controlled, one-per-init observations the
-- meta-dataset's train/test split is built from. Mixing them into
-- `experiments` would silently break every script that assumes exactly one
-- row per (task, init) at the fixed search-space point build_meta_dataset.py
-- established.
CREATE TABLE IF NOT EXISTS search_trials (
    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    run_label TEXT NOT NULL,
    seed INTEGER NOT NULL,
    arm TEXT NOT NULL,
    trial_number INTEGER NOT NULL,
    learning_rate REAL NOT NULL,
    init_method TEXT NOT NULL,
    steps_to_threshold REAL NOT NULL
);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def experiment_exists(seed: int, init_method: str) -> bool:
    """Lets build_meta_dataset.py grow the meta-dataset incrementally
    (bigger --n-tasks, same seed_offset) without ever needing to delete
    data/meta_dataset.db -- which would silently wipe the gate_evaluations
    and search_trials history too (found the hard way: rebuilding for a
    larger meta-dataset once already did exactly that)."""
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM experiments WHERE seed = ? AND json_extract(training_json, '$.init_method') = ? LIMIT 1",
            (seed, init_method),
        ).fetchone()
        return row is not None


def record_experiment(
    *,
    split: str,
    seed: int,
    mode: str,
    model_features: dict,
    task_features: dict,
    hardware_features: dict,
    regime: dict,
    training_config: dict,
    outcome,  # precog.modes.TrainResult
    zero_cost_features: dict | None = None,
) -> int:
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO experiments (
                split, seed, mode, model_json, task_json, hardware_json, regime_json,
                training_json, zero_cost_json, initial_loss, final_loss, steps_to_threshold,
                converged, diverged, wall_clock_s, delta_w_norm
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                split,
                seed,
                mode,
                json.dumps(model_features),
                json.dumps(task_features),
                json.dumps(hardware_features),
                json.dumps(regime),
                json.dumps(training_config),
                json.dumps(zero_cost_features) if zero_cost_features else None,
                outcome.initial_loss,
                outcome.final_loss,
                outcome.steps_to_threshold,
                int(outcome.converged),
                int(outcome.diverged),
                outcome.wall_clock_s,
                outcome.delta_w_norm,
            ),
        )
        return cursor.lastrowid


def record_search_trial(
    *,
    run_label: str,
    seed: int,
    arm: str,
    trial_number: int,
    learning_rate: float,
    init_method: str,
    steps_to_threshold: float,
) -> int:
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO search_trials (
                run_label, seed, arm, trial_number, learning_rate, init_method, steps_to_threshold
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_label, seed, arm, trial_number, learning_rate, init_method, steps_to_threshold),
        )
        return cursor.lastrowid


def load_search_trials():
    import pandas as pd

    with connect() as conn:
        return pd.read_sql_query("SELECT * FROM search_trials ORDER BY timestamp", conn)


def record_gate_evaluation(
    *,
    generation: str,
    gate_number: int,
    metric_name: str,
    metric_value: float,
    threshold: float,
    n_samples: int | None = None,
    notes: str | None = None,
) -> int:
    passed = abs(metric_value) >= threshold if metric_value is not None else False
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO gate_evaluations (
                generation, gate_number, metric_name, metric_value, threshold,
                passed, n_samples, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (generation, gate_number, metric_name, metric_value, threshold, int(passed), n_samples, notes),
        )
        return cursor.lastrowid


def load_gate_history():
    import pandas as pd

    with connect() as conn:
        return pd.read_sql_query("SELECT * FROM gate_evaluations ORDER BY timestamp", conn)


def load_dataframe(split: str | None = None):
    import pandas as pd

    query = "SELECT * FROM experiments"
    params = ()
    if split is not None:
        query += " WHERE split = ?"
        params = (split,)
    with connect() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    for col in ("model_json", "task_json", "hardware_json", "regime_json", "training_json", "zero_cost_json"):
        expanded = pd.json_normalize(df[col].apply(lambda s: json.loads(s) if s else {}))
        expanded.columns = [f"{col.removesuffix('_json')}.{c}" for c in expanded.columns]
        df = pd.concat([df.drop(columns=[col]), expanded], axis=1)
    return df
