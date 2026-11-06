"""Persistent research ledger: hypotheses, experiments and review verdicts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round INTEGER NOT NULL,
    text TEXT NOT NULL,
    evidence TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round INTEGER NOT NULL,
    tool TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    findings TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def record_hypothesis(self, round_index: int, text: str, evidence: str = "") -> None:
        self._connection.execute(
            "INSERT INTO hypotheses (round, text, evidence) VALUES (?, ?, ?)",
            (round_index, text, evidence),
        )
        self._connection.commit()

    def record_experiment(self, round_index: int, tool: str, payload: dict) -> None:
        self._connection.execute(
            "INSERT INTO experiments (round, tool, payload) VALUES (?, ?, ?)",
            (round_index, tool, json.dumps(payload, ensure_ascii=False, allow_nan=False)),
        )
        self._connection.commit()

    def record_review(self, round_index: int, verdict: str, findings: list[dict]) -> None:
        self._connection.execute(
            "INSERT INTO reviews (round, verdict, findings) VALUES (?, ?, ?)",
            (round_index, verdict, json.dumps(findings, ensure_ascii=False)),
        )
        self._connection.commit()

    def hypotheses(self) -> list[dict]:
        rows = self._connection.execute(
            "SELECT round, text, evidence FROM hypotheses ORDER BY id"
        ).fetchall()
        return [{"round": r[0], "text": r[1], "evidence": r[2]} for r in rows]

    def experiments(self) -> list[dict]:
        rows = self._connection.execute(
            "SELECT round, tool, payload FROM experiments ORDER BY id"
        ).fetchall()
        return [
            {"round": r[0], "tool": r[1], "payload": json.loads(r[2])} for r in rows
        ]

    def reviews(self) -> list[dict]:
        rows = self._connection.execute(
            "SELECT round, verdict, findings FROM reviews ORDER BY id"
        ).fetchall()
        return [
            {"round": r[0], "verdict": r[1], "findings": json.loads(r[2])}
            for r in rows
        ]
