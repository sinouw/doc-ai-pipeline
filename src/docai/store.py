"""SQLite persistence for processed documents, per-field review state and reviewer corrections."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import DocumentResult, FieldStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    method TEXT NOT NULL,
    language TEXT,
    pages INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fields (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value_json TEXT,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    issues_json TEXT NOT NULL,
    signals_json TEXT NOT NULL,
    UNIQUE (document_id, name)
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY,
    field_id INTEGER NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    old_value_json TEXT,
    new_value_json TEXT,
    action TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fields_status ON fields(status, confidence);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.pages_dir = data_dir / "pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(data_dir / "docai.sqlite3", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self._conn:
            yield self._conn

    def save(self, result: DocumentResult, page_pngs: list[bytes]) -> int:
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT INTO documents (filename, method, language, pages, created_at, result_json) VALUES (?, ?, ?, ?, ?, ?)",
                (result.source, result.method, result.language, result.pages, _now(), result.model_dump_json()),
            )
            doc_id = int(cur.lastrowid or 0)
            conn.executemany(
                "INSERT INTO fields (document_id, name, value_json, confidence, status, issues_json, signals_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        doc_id,
                        f.name,
                        json.dumps(f.value, ensure_ascii=False),
                        f.confidence,
                        f.status.value,
                        json.dumps(f.issues, ensure_ascii=False),
                        json.dumps(f.signals),
                    )
                    for f in result.fields
                ],
            )
        for i, png in enumerate(page_pngs):
            (self.pages_dir / f"{doc_id}-{i}.png").write_bytes(png)
        return doc_id

    def page_path(self, doc_id: int, page: int) -> Path | None:
        path = self.pages_dir / f"{doc_id}-{page}.png"
        return path if path.is_file() else None

    @staticmethod
    def _field(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "document_id": row["document_id"],
            "name": row["name"],
            "value": json.loads(row["value_json"]) if row["value_json"] is not None else None,
            "confidence": row["confidence"],
            "status": row["status"],
            "issues": json.loads(row["issues_json"]),
            "signals": json.loads(row["signals_json"]),
        }

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT d.id, d.filename, d.method, d.language, d.pages, d.created_at,
                      SUM(f.status = 'needs_review') AS pending
               FROM documents d LEFT JOIN fields f ON f.document_id = d.id
               GROUP BY d.id ORDER BY d.id DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_document(self, doc_id: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        result = json.loads(row["result_json"])
        fields = self._conn.execute("SELECT * FROM fields WHERE document_id = ? ORDER BY id", (doc_id,)).fetchall()
        return {
            "id": row["id"],
            "filename": row["filename"],
            "method": row["method"],
            "language": row["language"],
            "pages": row["pages"],
            "created_at": row["created_at"],
            "issues": result["issues"],
            "usage": result["usage"],
            "timings_ms": result["timings_ms"],
            "fields": [self._field(f) for f in fields],
        }

    def review_queue(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT f.*, d.filename FROM fields f JOIN documents d ON d.id = f.document_id
               WHERE f.status = ? ORDER BY f.confidence ASC, f.id ASC LIMIT ?""",
            (FieldStatus.NEEDS_REVIEW.value, limit),
        ).fetchall()
        return [{**self._field(r), "filename": r["filename"]} for r in rows]

    def get_field(self, field_id: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM fields WHERE id = ?", (field_id,)).fetchone()
        return self._field(row) if row else None

    def resolve(self, field_id: int, new_value: Any, reviewer: str, action: str) -> dict[str, Any]:
        """Record a reviewer decision. `action` is 'correct' (new value) or 'confirm' (keep value)."""
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM fields WHERE id = ?", (field_id,)).fetchone()
            if row is None:
                raise KeyError(field_id)
            new_json = json.dumps(new_value, ensure_ascii=False) if action == "correct" else row["value_json"]
            conn.execute(
                "INSERT INTO corrections (field_id, document_id, name, old_value_json, new_value_json, action, reviewer, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (field_id, row["document_id"], row["name"], row["value_json"], new_json, action, reviewer, _now()),
            )
            conn.execute(
                "UPDATE fields SET value_json = ?, status = ?, confidence = 1.0, issues_json = '[]' WHERE id = ?",
                (new_json, FieldStatus.REVIEWED.value, field_id),
            )
        field = self.get_field(field_id)
        assert field is not None
        return field

    def corrections(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM corrections ORDER BY id").fetchall()
        return [
            {
                **{k: r[k] for k in ("id", "field_id", "document_id", "name", "action", "reviewer", "created_at")},
                "old_value": json.loads(r["old_value_json"]) if r["old_value_json"] else None,
                "new_value": json.loads(r["new_value_json"]) if r["new_value_json"] else None,
            }
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
