"""SQLite persistence for classified news items."""

from __future__ import annotations

import json
import sqlite3
import time
from calendar import timegm
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS news_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    link TEXT NOT NULL UNIQUE,
    title TEXT,
    body_excerpt TEXT,
    published_at REAL,
    classification TEXT,
    entities_json TEXT,
    sentiment_score REAL,
    reasoning TEXT,
    analyzed_at REAL NOT NULL,
    error TEXT,
    raw_response TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_analyses_analyzed ON news_analyses(analyzed_at);
CREATE INDEX IF NOT EXISTS idx_news_analyses_published ON news_analyses(published_at);
"""


@dataclass(frozen=True)
class StoredNewsRow:
    source: str
    link: str
    title: str
    body_excerpt: str
    published_at: float | None
    classification: str
    entities: tuple[str, ...]
    sentiment_score: float
    reasoning: str
    analyzed_at: float
    error: str | None


def published_tuple_to_unix(tup: tuple | None) -> float | None:
    """feedparser time struct -> unix seconds UTC."""
    if tup is None:
        return None
    try:
        return float(timegm(tup))
    except (TypeError, ValueError):
        return None


class NewsSQLiteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def link_exists(self, link: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("SELECT 1 FROM news_analyses WHERE link = ? LIMIT 1", (link,))
            return cur.fetchone() is not None

    def insert_success(
        self,
        *,
        source: str,
        link: str,
        title: str,
        body_excerpt: str,
        published_at: float | None,
        classification: str,
        entities: Sequence[str],
        sentiment_score: float,
        reasoning: str,
        raw_response: str | None = None,
    ) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO news_analyses (
                    source, link, title, body_excerpt, published_at,
                    classification, entities_json, sentiment_score, reasoning,
                    analyzed_at, error, raw_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(link) DO NOTHING
                """,
                (
                    source,
                    link,
                    title,
                    body_excerpt,
                    published_at,
                    classification,
                    json.dumps(list(entities)),
                    sentiment_score,
                    reasoning,
                    now,
                    raw_response,
                ),
            )

    def insert_error(
        self,
        *,
        source: str,
        link: str,
        title: str,
        body_excerpt: str,
        published_at: float | None,
        error: str,
    ) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO news_analyses (
                    source, link, title, body_excerpt, published_at,
                    classification, entities_json, sentiment_score, reasoning,
                    analyzed_at, error, raw_response
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?, NULL)
                ON CONFLICT(link) DO NOTHING
                """,
                (source, link, title, body_excerpt, published_at, now, error[:2000]),
            )

    def fetch_since(self, since_unix: float) -> list[StoredNewsRow]:
        """Rows with successful classification and analyzed_at >= since (or published in window)."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT source, link, title, body_excerpt, published_at, classification,
                       entities_json, sentiment_score, reasoning, analyzed_at, error
                FROM news_analyses
                WHERE error IS NULL AND classification IS NOT NULL
                  AND (COALESCE(published_at, analyzed_at) >= ?)
                ORDER BY COALESCE(published_at, analyzed_at) DESC
                """,
                (since_unix,),
            )
            rows = cur.fetchall()
        out: list[StoredNewsRow] = []
        for r in rows:
            ent_raw = r["entities_json"]
            try:
                elist = json.loads(ent_raw) if ent_raw else []
            except json.JSONDecodeError:
                elist = []
            if not isinstance(elist, list):
                elist = []
            ents = tuple(str(x).upper() for x in elist if isinstance(x, str))
            out.append(
                StoredNewsRow(
                    source=str(r["source"]),
                    link=str(r["link"]),
                    title=str(r["title"] or ""),
                    body_excerpt=str(r["body_excerpt"] or ""),
                    published_at=float(r["published_at"]) if r["published_at"] is not None else None,
                    classification=str(r["classification"]),
                    entities=ents,
                    sentiment_score=float(r["sentiment_score"] or 0.0),
                    reasoning=str(r["reasoning"] or ""),
                    analyzed_at=float(r["analyzed_at"]),
                    error=str(r["error"]) if r["error"] else None,
                )
            )
        return out
