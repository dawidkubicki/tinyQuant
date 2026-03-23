"""RSS fetch and text extraction (ported from legacy news.py)."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import feedparser

TAG_RE = re.compile(r"<[^>]+>")

DEFAULT_FEEDS: dict[str, str] = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "cryptoslate": "https://cryptoslate.com/feed/",
}


def strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1>", " ", html)
    text = TAG_RE.sub(" ", html)
    return unescape(re.sub(r"\s+", " ", text)).strip()


def body_from_feed_entry(entry: feedparser.FeedParserDict) -> str:
    """Prefer full RSS content when present; otherwise summary/description."""
    summary = strip_html(getattr(entry, "summary", "") or "")
    longest = ""
    for block in getattr(entry, "content", None) or []:
        if isinstance(block, dict):
            val = (block.get("value") or "").strip()
        else:
            val = str(block).strip()
        if not val:
            continue
        t = strip_html(val)
        if len(t) > len(longest):
            longest = t
    if len(longest) > len(summary) + 80:
        return longest
    return longest or summary


def rss_link_teasers(feed_url: str, timeout: float = 20.0, user_agent: str | None = None) -> dict[str, str]:
    """
    Map item link -> <description> text from raw RSS XML.

    feedparser sometimes drops the RSS description when <content:encoded/> is
    present but empty (e.g. CoinDesk). One request per feed fixes teasers.
    """
    ua = user_agent or "tinyQuant-news/0.1 (+https://github.com/) research"
    req = Request(feed_url, headers={"User-Agent": ua})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (HTTPError, URLError, TimeoutError, OSError):
        return {}
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return {}
    out: dict[str, str] = {}
    channel = root.find("channel")
    if channel is None:
        return out
    for item in channel.findall("item"):
        link_el = item.find("link")
        desc_el = item.find("description")
        if link_el is None or desc_el is None:
            continue
        link = (link_el.text or "").strip()
        desc = (desc_el.text or "").strip()
        if link and desc:
            out[link] = desc
    return out


def merge_teaser(body: str, teaser: str) -> str:
    t = strip_html(teaser) if teaser else ""
    return t if len(t) > len(body) else body


def fetch_ld_json_article_text(url: str, timeout: float = 20.0, user_agent: str | None = None) -> str:
    """
    Best-effort: parse application/ld+json on the page for article text.
    Works when publishers embed articleBody (many blogs); often not full on SPAs.
    """
    ua = user_agent or "tinyQuant-news/0.1 (+https://github.com/) research"
    req = Request(url, headers={"User-Agent": ua})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError, OSError):
        return ""

    best = ""
    for m in re.finditer(
        r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
        raw,
        re.DOTALL | re.IGNORECASE,
    ):
        chunk = m.group(1).strip()
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        objs = data if isinstance(data, list) else [data]
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            t = obj.get("@type")
            types = t if isinstance(t, list) else ([t] if t else [])
            if not any(x in ("NewsArticle", "Article", "BlogPosting") for x in types):
                continue
            for key in ("articleBody", "text"):
                val = obj.get(key)
                if isinstance(val, str) and len(val) > len(best):
                    best = val
            ab = obj.get("abstract")
            if isinstance(ab, str) and len(ab) > len(best):
                best = ab
            desc = obj.get("description")
            if isinstance(desc, str) and len(desc) > len(best):
                best = desc
    return best.strip()


@dataclass(frozen=True)
class RssNewsItem:
    """One RSS entry ready for LLM classification."""

    source: str
    title: str
    link: str
    text_for_model: str
    published_parsed: tuple | None  # feedparser time tuple or None


def build_item_text(
    entry: feedparser.FeedParserDict,
    *,
    teasers: dict[str, str],
    max_body_chars: int,
    fetch_ld_json: bool,
    rss_timeout: float,
    user_agent: str | None = None,
) -> str:
    body = body_from_feed_entry(entry)
    link = getattr(entry, "link", "") or ""
    body = merge_teaser(body, teasers.get(link, ""))
    if fetch_ld_json and link:
        extra = fetch_ld_json_article_text(link, timeout=rss_timeout, user_agent=user_agent)
        if len(extra) > len(body):
            body = extra
    if max_body_chars > 0 and len(body) > max_body_chars:
        body = body[:max_body_chars].rstrip() + "…"
    return body


def fetch_feed_items(
    source_name: str,
    feed_url: str,
    *,
    limit: int,
    max_body_chars: int,
    fetch_ld_json: bool,
    rss_timeout: float,
    user_agent: str | None = None,
) -> list[RssNewsItem]:
    """Parse feed and return up to `limit` items with merged body text."""
    feed = feedparser.parse(feed_url)
    teasers = rss_link_teasers(feed_url, timeout=rss_timeout, user_agent=user_agent)
    out: list[RssNewsItem] = []
    for entry in feed.entries[:limit]:
        title = getattr(entry, "title", "") or ""
        link = getattr(entry, "link", "") or ""
        text = build_item_text(
            entry,
            teasers=teasers,
            max_body_chars=max_body_chars,
            fetch_ld_json=fetch_ld_json,
            rss_timeout=rss_timeout,
            user_agent=user_agent,
        )
        published = getattr(entry, "published_parsed", None)
        out.append(
            RssNewsItem(
                source=source_name,
                title=title.strip(),
                link=link.strip(),
                text_for_model=text.strip(),
                published_parsed=published if published else None,
            )
        )
    return out
