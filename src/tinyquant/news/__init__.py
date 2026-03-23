"""RSS news ingestion, Ollama classification, SQLite storage, and sentiment aggregation."""

from __future__ import annotations

from tinyquant.news.worker import run_news_loop, sync_news_once

__all__ = ["sync_news_once", "run_news_loop"]
