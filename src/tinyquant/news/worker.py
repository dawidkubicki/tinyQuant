"""Background-friendly RSS fetch + Ollama classification + SQLite upsert."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass

from tinyquant.config.schema import SentimentNewsConfig
from tinyquant.news.ollama_client import classify_news_text
from tinyquant.news.rss_ingest import RssNewsItem, fetch_feed_items
from tinyquant.news.store_sqlite import NewsSQLiteStore, published_tuple_to_unix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsSyncStats:
    feeds_processed: int
    items_seen: int
    items_classified: int
    items_skipped_existing: int
    items_failed: int


def _user_prompt(item: RssNewsItem) -> str:
    parts = []
    if item.title:
        parts.append(f"Title: {item.title}")
    if item.text_for_model:
        parts.append(f"Body: {item.text_for_model}")
    elif not parts:
        parts.append("(empty)")
    return "\n\n".join(parts)


def classify_with_retries(
    text: str,
    news_cfg: SentimentNewsConfig,
    *,
    ollama_host: str | None = None,
) -> tuple[object | None, str | None]:
    """Returns (NewsClassificationResult | None, raw_json_or_error)."""
    host = (ollama_host or os.environ.get("OLLAMA_HOST") or news_cfg.ollama_host).strip()
    last_err: str | None = None
    attempts = max(1, news_cfg.ollama_max_retries + 1)
    for _ in range(attempts):
        res = classify_news_text(
            text,
            model=news_cfg.ollama_model,
            host=host,
            timeout=news_cfg.ollama_timeout_seconds,
        )
        if res is not None:
            raw = json.dumps(
                {
                    "classification": res.classification,
                    "entities": list(res.entities),
                    "sentiment_score": res.sentiment_score,
                    "reasoning": res.reasoning,
                }
            )
            return res, raw
        last_err = "ollama_returned_null_or_invalid_json"
        time.sleep(0.5)
    return None, last_err


def sync_news_once(news_cfg: SentimentNewsConfig) -> NewsSyncStats:
    store = NewsSQLiteStore(news_cfg.sqlite_path)
    store.init_schema()

    feeds_processed = 0
    items_seen = 0
    items_classified = 0
    items_skipped = 0
    items_failed = 0

    for feed in news_cfg.feeds:
        feeds_processed += 1
        items = fetch_feed_items(
            feed.name,
            feed.url,
            limit=news_cfg.limit_per_feed,
            max_body_chars=news_cfg.max_body_chars,
            fetch_ld_json=news_cfg.fetch_ld_json,
            rss_timeout=news_cfg.rss_timeout_seconds,
            user_agent=news_cfg.user_agent,
        )
        for item in items:
            items_seen += 1
            if not item.link:
                items_failed += 1
                continue
            if store.link_exists(item.link):
                items_skipped += 1
                continue

            prompt = _user_prompt(item)
            if not prompt.strip():
                store.insert_error(
                    source=item.source,
                    link=item.link,
                    title=item.title,
                    body_excerpt="",
                    published_at=published_tuple_to_unix(item.published_parsed),
                    error="empty_prompt",
                )
                items_failed += 1
                continue

            result, raw_or_err = classify_with_retries(prompt, news_cfg)
            pub = published_tuple_to_unix(item.published_parsed)

            if result is not None:
                excerpt = item.text_for_model[:2000] if item.text_for_model else ""
                store.insert_success(
                    source=item.source,
                    link=item.link,
                    title=item.title,
                    body_excerpt=excerpt,
                    published_at=pub,
                    classification=result.classification,
                    entities=result.entities,
                    sentiment_score=result.sentiment_score,
                    reasoning=result.reasoning,
                    raw_response=raw_or_err,
                )
                items_classified += 1
            else:
                store.insert_error(
                    source=item.source,
                    link=item.link,
                    title=item.title,
                    body_excerpt=item.text_for_model[:2000] if item.text_for_model else "",
                    published_at=pub,
                    error=raw_or_err or "unknown_error",
                )
                items_failed += 1

    return NewsSyncStats(
        feeds_processed=feeds_processed,
        items_seen=items_seen,
        items_classified=items_classified,
        items_skipped_existing=items_skipped,
        items_failed=items_failed,
    )


def run_news_loop(
    news_cfg: SentimentNewsConfig,
    *,
    stop_event: threading.Event | None = None,
) -> None:
    """Sleep `poll_interval_seconds` between syncs until stop_event is set."""
    ev = stop_event
    while True:
        try:
            stats = sync_news_once(news_cfg)
            logger.info(
                "News sync: classified=%s skipped=%s failed=%s feeds=%s",
                stats.items_classified,
                stats.items_skipped_existing,
                stats.items_failed,
                stats.feeds_processed,
            )
        except Exception:
            logger.exception("News sync crashed")
        if ev is not None and ev.is_set():
            break
        interval = max(30.0, float(news_cfg.poll_interval_seconds))
        if ev is not None:
            if ev.wait(timeout=interval):
                break
        else:
            time.sleep(interval)
