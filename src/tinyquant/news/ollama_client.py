"""Ollama chat with strict JSON output for news classification."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

Classification = Literal["MACRO", "MICRO", "NOISE"]

SYSTEM_PROMPT = """You are a strict, quantitative financial analyst AI for a crypto hedge fund.
Your task is to read a news snippet and convert it into a structured JSON format.
You must classify the news into one of three categories:
1. "MACRO" - News affecting the whole market (e.g., Fed, wars, SEC, global liquidity, Bitcoin as a macro asset).
2. "MICRO" - News specific to individual altcoins or projects (e.g., network upgrades, token unlocks, airdrops).
3. "NOISE" - Clickbait, influencer drama, or irrelevant news.

Analyze the sentiment strictly from -1.0 (Extreme Panic/Bearish) to +1.0 (Euphoria/Bullish). 0.0 is neutral.

You must reply ONLY with a valid JSON object. Do not add any markdown, greetings, or explanations outside the JSON.

Expected JSON Schema:
{
  "classification": "MACRO" | "MICRO" | "NOISE",
  "entities": ["TICKER1", "TICKER2"],
  "sentiment_score": float,
  "reasoning": "One short sentence explaining the score."
}

Use empty [] for entities if MACRO or NOISE."""


@dataclass(frozen=True)
class NewsClassificationResult:
    classification: Classification
    entities: tuple[str, ...]
    sentiment_score: float
    reasoning: str


def _normalize_entities(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    for x in raw:
        if not isinstance(x, str):
            continue
        t = x.strip().upper()
        t = re.sub(r"[^A-Z0-9]", "", t)
        if t:
            out.append(t)
    return tuple(dict.fromkeys(out))  # dedupe preserve order


def parse_classification_json(content: str) -> NewsClassificationResult | None:
    """Parse and validate model JSON string. Returns None if invalid."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    cls = obj.get("classification")
    if cls not in ("MACRO", "MICRO", "NOISE"):
        return None
    score = obj.get("sentiment_score")
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    s = max(-1.0, min(1.0, s))
    reasoning = obj.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    entities = _normalize_entities(obj.get("entities"))
    if cls in ("MACRO", "NOISE"):
        entities = ()
    return NewsClassificationResult(
        classification=cls,
        entities=entities,
        sentiment_score=s,
        reasoning=reasoning.strip()[:500],
    )


def classify_news_text(
    news_text: str,
    *,
    model: str,
    host: str = "http://127.0.0.1:11434",
    timeout: float = 120.0,
) -> NewsClassificationResult | None:
    """
    Call Ollama chat API with JSON format. Returns None on failure.

    Uses optional dependency `ollama`; import lazily so installs without ollama still work.
    """
    try:
        from ollama import Client
    except ImportError:
        logger.error("ollama package not installed; pip install ollama")
        return None

    try:
        client = Client(host=host, timeout=timeout)
    except TypeError:
        client = Client(host=host)
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": news_text},
            ],
            format="json",
        )
    except Exception as e:
        logger.warning("Ollama chat failed: %s", e)
        return None

    if isinstance(response, dict):
        msg = response.get("message")
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = getattr(msg, "content", "") if msg is not None else ""
    else:
        msg = getattr(response, "message", None)
        content = getattr(msg, "content", "") if msg is not None else ""
    if not isinstance(content, str):
        return None
    return parse_classification_json(content.strip())
