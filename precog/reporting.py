"""Exports the meta-dataset (docs.md §12) and gate history (§17) into
`results/` as plain CSV + Markdown -- exploitable outside SQLite (Excel,
pandas, a text editor, grep) by anyone who doesn't want to query the DB
directly, and a durable, timestamped record of every test run rather than
whatever happened to be left in a terminal scrollback.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from precog.experiment_db import load_dataframe, load_gate_history, load_search_trials

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
REPORTS_DIR = RESULTS_DIR / "reports"


def export_csv_snapshots() -> None:
    """Refreshes results/experiments.csv, results/gate_evaluations.csv and
    results/search_trials.csv from the current state of the meta-dataset
    (§12/§17)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = load_dataframe()
    experiments.to_csv(RESULTS_DIR / "experiments.csv", index=False)

    gates = load_gate_history()
    gates.to_csv(RESULTS_DIR / "gate_evaluations.csv", index=False)

    search_trials = load_search_trials()
    search_trials.to_csv(RESULTS_DIR / "search_trials.csv", index=False)


def write_report(slug: str, title: str, body_markdown: str) -> Path:
    """Writes a timestamped Markdown report to results/reports/ and returns
    its path. `body_markdown` is the caller's fully-formed report content
    (methodology, parameters, metrics, verdict) -- this function only
    handles the timestamped filename and the shared header."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = REPORTS_DIR / f"{timestamp}_{slug}.md"
    header = f"# {title}\n\n_Generated {timestamp} (UTC)_\n\n"
    path.write_text(header + body_markdown)
    return path
