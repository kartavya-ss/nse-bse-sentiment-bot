import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    channel TEXT,
                    timestamp TEXT,
                    published_at TEXT,
                    snippet TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    feedback_text TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_cache (
                    cache_key TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    response_json TEXT NOT NULL
                )
                """
            )

    def upsert_document(self, doc: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, source_type, title, channel, timestamp, published_at, snippet, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_type=excluded.source_type,
                    title=excluded.title,
                    channel=excluded.channel,
                    timestamp=excluded.timestamp,
                    published_at=excluded.published_at,
                    snippet=excluded.snippet,
                    metadata_json=excluded.metadata_json
                """,
                (
                    doc["id"],
                    doc["source_type"],
                    doc["title"],
                    doc.get("channel"),
                    doc.get("timestamp"),
                    doc.get("published_at"),
                    doc["snippet"],
                    json.dumps(doc.get("metadata", {}), default=str),
                ),
            )

    def get_documents_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM documents WHERE id IN ({placeholders})", ids
            ).fetchall()
        return [dict(row) for row in rows]

    def insert_feedback(
        self, question: str, answer: str, rating: int, feedback_text: str | None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (created_at, question, answer, rating, feedback_text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    question,
                    answer,
                    rating,
                    feedback_text,
                ),
            )

    def recent_feedback(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chat_cache(self, cache_key: str, ttl_seconds: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cache_key, created_at, response_json FROM chat_cache WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        created_at = datetime.fromisoformat(row["created_at"])
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - created_at > timedelta(seconds=ttl_seconds):
            self.delete_chat_cache(cache_key)
            return None
        return json.loads(row["response_json"])

    def upsert_chat_cache(self, cache_key: str, response: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_cache (cache_key, created_at, response_json)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    created_at=excluded.created_at,
                    response_json=excluded.response_json
                """,
                (
                    cache_key,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(response, default=str),
                ),
            )

    def delete_chat_cache(self, cache_key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_cache WHERE cache_key=?", (cache_key,))
