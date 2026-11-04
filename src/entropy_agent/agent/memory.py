"""SQLite experiment memory: every trial survives, including failures.

The schema is deliberately plain so the table can be inspected with any SQLite
client and every reported number can be recomputed from a trial row.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    budget_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    champion_json TEXT,
    test_json TEXT
);
CREATE TABLE IF NOT EXISTS trials (
    run_id TEXT NOT NULL,
    arm TEXT NOT NULL,
    trial_index INTEGER NOT NULL,
    round_index INTEGER NOT NULL,
    candidate_id TEXT NOT NULL,
    method TEXT NOT NULL,
    family TEXT NOT NULL,
    method_parameters TEXT NOT NULL,
    classifier TEXT NOT NULL,
    classifier_backend TEXT NOT NULL,
    classifier_parameters TEXT NOT NULL,
    feature_key TEXT NOT NULL,
    cache_hit INTEGER NOT NULL,
    status TEXT NOT NULL,
    failure TEXT,
    val_accuracy REAL,
    val_macro_f1 REAL,
    extract_seconds REAL NOT NULL,
    fit_seconds REAL NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, arm, trial_index)
);
CREATE INDEX IF NOT EXISTS trials_by_run ON trials (run_id, arm, trial_index);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentMemory:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ExperimentMemory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start_run(
        self,
        run_id: str,
        dataset_id: str,
        profile: dict,
        budget: dict,
    ) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO runs (run_id, dataset_id, profile_json, budget_json, "
            "started_at, finished_at, champion_json, test_json) "
            "VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)",
            (run_id, dataset_id, json.dumps(profile, sort_keys=True),
             json.dumps(budget, sort_keys=True), _now()),
        )
        self._connection.execute("DELETE FROM trials WHERE run_id = ?", (run_id,))
        self._connection.commit()

    def record_trial(
        self,
        *,
        run_id: str,
        arm: str,
        trial_index: int,
        round_index: int,
        candidate: dict,
        feature_key: str,
        cache_hit: bool,
        status: str,
        failure: str | None,
        val_accuracy: float | None,
        val_macro_f1: float | None,
        extract_seconds: float,
        fit_seconds: float,
    ) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO trials ("
            "run_id, arm, trial_index, round_index, candidate_id, method, family, "
            "method_parameters, classifier, classifier_backend, classifier_parameters, "
            "feature_key, cache_hit, status, failure, val_accuracy, val_macro_f1, "
            "extract_seconds, fit_seconds, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, arm, trial_index, round_index,
                candidate["candidate_id"], candidate["method"], candidate["family"],
                json.dumps(candidate["method_parameters"], sort_keys=True),
                candidate["classifier"], candidate["classifier_backend"],
                json.dumps(candidate["classifier_parameters"], sort_keys=True),
                feature_key, int(cache_hit), status, failure,
                val_accuracy, val_macro_f1, float(extract_seconds), float(fit_seconds),
                _now(),
            ),
        )
        self._connection.commit()

    def finish_run(self, run_id: str, champion: dict | None, test: dict | None) -> None:
        self._connection.execute(
            "UPDATE runs SET finished_at = ?, champion_json = ?, test_json = ? WHERE run_id = ?",
            (_now(), json.dumps(champion, sort_keys=True) if champion else None,
             json.dumps(test, sort_keys=True) if test else None, run_id),
        )
        self._connection.commit()

    def trials(self, run_id: str, arm: str | None = None) -> list[dict]:
        query = "SELECT * FROM trials WHERE run_id = ?"
        parameters: list[object] = [run_id]
        if arm is not None:
            query += " AND arm = ?"
            parameters.append(arm)
        query += " ORDER BY arm, trial_index"
        rows = self._connection.execute(query, parameters).fetchall()
        return [_trial_row(row) for row in rows]

    def run(self, run_id: str) -> dict | None:
        row = self._connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "dataset_id": row["dataset_id"],
            "profile": json.loads(row["profile_json"]),
            "budget": json.loads(row["budget_json"]),
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "champion": json.loads(row["champion_json"]) if row["champion_json"] else None,
            "test": json.loads(row["test_json"]) if row["test_json"] else None,
        }


def _trial_row(row: sqlite3.Row) -> dict:
    return {
        "run_id": row["run_id"],
        "arm": row["arm"],
        "trial_index": row["trial_index"],
        "round_index": row["round_index"],
        "candidate_id": row["candidate_id"],
        "method": row["method"],
        "family": row["family"],
        "method_parameters": json.loads(row["method_parameters"]),
        "classifier": row["classifier"],
        "classifier_backend": row["classifier_backend"],
        "classifier_parameters": json.loads(row["classifier_parameters"]),
        "feature_key": row["feature_key"],
        "cache_hit": bool(row["cache_hit"]),
        "status": row["status"],
        "failure": row["failure"],
        "val_accuracy": row["val_accuracy"],
        "val_macro_f1": row["val_macro_f1"],
        "extract_seconds": row["extract_seconds"],
        "fit_seconds": row["fit_seconds"],
        "created_at": row["created_at"],
    }
