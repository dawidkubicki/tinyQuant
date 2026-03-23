from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from tinyquant.news.aggregation import aggregate_news_for_universe, decay_factor, effective_sentiment
from tinyquant.news.ollama_client import parse_classification_json
from tinyquant.news.store_sqlite import NewsSQLiteStore
from tinyquant.orchestration.h4_cycle import run_h4_cycle


def test_parse_classification_json_valid() -> None:
    raw = json.dumps(
        {
            "classification": "MICRO",
            "entities": ["sol", "BP"],
            "sentiment_score": 0.6,
            "reasoning": "Ecosystem expansion.",
        }
    )
    r = parse_classification_json(raw)
    assert r is not None
    assert r.classification == "MICRO"
    assert r.entities == ("SOL", "BP")
    assert abs(r.sentiment_score - 0.6) < 1e-9


def test_parse_classification_json_rejects_invalid() -> None:
    assert parse_classification_json("not json") is None
    assert parse_classification_json('{"classification": "MEH"}') is None
    assert parse_classification_json('{"classification": "MACRO", "sentiment_score": "x"}') is None


def test_parse_macro_clears_entities() -> None:
    raw = json.dumps(
        {
            "classification": "MACRO",
            "entities": ["BTC"],
            "sentiment_score": -0.2,
            "reasoning": "Risk-off.",
        }
    )
    r = parse_classification_json(raw)
    assert r is not None
    assert r.entities == ()


def test_decay_factor_edges() -> None:
    assert decay_factor(0.0, 0.9) == pytest.approx(1.0)
    assert decay_factor(1.0, 0.9) == pytest.approx(0.9)
    assert decay_factor(24.0, 0.9) == pytest.approx(0.9**24)
    assert decay_factor(1.0, 1.0) == 1.0


def test_effective_sentiment() -> None:
    assert effective_sentiment(1.0, 0.0, 0.9) == pytest.approx(1.0)
    assert effective_sentiment(1.0, 1.0, 0.9) == pytest.approx(0.9)


def test_sqlite_dedup_link(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    store = NewsSQLiteStore(db)
    store.init_schema()
    store.insert_success(
        source="t",
        link="https://example.com/a",
        title="T",
        body_excerpt="b",
        published_at=1000.0,
        classification="NOISE",
        entities=(),
        sentiment_score=0.0,
        reasoning="n",
    )
    store.insert_success(
        source="t",
        link="https://example.com/a",
        title="T2",
        body_excerpt="b2",
        published_at=2000.0,
        classification="MICRO",
        entities=("SOL",),
        sentiment_score=0.5,
        reasoning="x",
    )
    conn = sqlite3.connect(str(db))
    n = conn.execute("SELECT COUNT(*) FROM news_analyses").fetchone()[0]
    assert n == 1


def test_aggregate_micro_and_macro(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    store = NewsSQLiteStore(db)
    store.init_schema()
    now = 1_700_000_000.0
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        INSERT INTO news_analyses (
            source, link, title, body_excerpt, published_at,
            classification, entities_json, sentiment_score, reasoning,
            analyzed_at, error, raw_response
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        (
            "x",
            "https://e/1",
            "t",
            "",
            now - 3600.0,
            "MICRO",
            json.dumps(["ALT1"]),
            0.8,
            "r",
            now,
        ),
    )
    conn.execute(
        """
        INSERT INTO news_analyses (
            source, link, title, body_excerpt, published_at,
            classification, entities_json, sentiment_score, reasoning,
            analyzed_at, error, raw_response
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        (
            "x",
            "https://e/2",
            "t2",
            "",
            now - 7200.0,
            "MACRO",
            json.dumps([]),
            1.0,
            "m",
            now,
        ),
    )
    conn.commit()
    conn.close()

    symbols = ["ALT1/USD:USD", "ALT2/USD:USD"]
    per_map, macro, _dbg = aggregate_news_for_universe(
        store,
        symbols,
        now=now,
        window_hours=24.0,
        decay_base=0.9,
        macro_blend_into_tokens=0.0,
    )
    assert abs(per_map[symbols[0]] - 0.8 * (0.9**1.0)) < 1e-6
    assert abs(macro - 1.0 * (0.9**2.0)) < 1e-6
    assert abs(per_map[symbols[1]]) < 1e-9


def test_h4_cycle_sentiment_enabled_with_seed_db(tmp_path: Path, strategy_config) -> None:
    db = tmp_path / "news.db"
    store = NewsSQLiteStore(db)
    store.init_schema()
    now = time.time()
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        INSERT INTO news_analyses (
            source, link, title, body_excerpt, published_at,
            classification, entities_json, sentiment_score, reasoning,
            analyzed_at, error, raw_response
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        (
            "x",
            "https://e/sol",
            "t",
            "",
            now,
            "MICRO",
            json.dumps(["ALT1"]),
            0.5,
            "r",
            now,
        ),
    )
    conn.commit()
    conn.close()

    sent = strategy_config.sentiment.model_copy(
        update={"enabled": True, "news": strategy_config.sentiment.news.model_copy(update={"sqlite_path": str(db)})}
    )
    cfg = strategy_config.model_copy(update={"sentiment": sent})
    out = run_h4_cycle(cfg, equity_usd=10_000.0, synthetic=True)
    assert out["sentiment"]["debug"]["mode"] == "news_db"
    alt_scores = [v for k, v in out["sentiment"]["per_tradable"].items() if k.startswith("ALT1")]
    assert alt_scores and abs(alt_scores[0]) > 1e-6
