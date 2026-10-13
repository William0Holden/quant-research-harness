"""Self-hosted news collector: broad market RSS + rotating per-ticker Yahoo RSS.

PIT: knowledge_time basis is fetched_at (crawl time). Claimed publish times are
stored for reference only. Politeness: one pass per cron run (hourly), capped
article-body fetches, 0.5s sleep between body fetches, honest UA. Writes ONLY the
corpus DB (never qts.duckdb); shell-level flock on outputs/.qts-corpus.lock.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests

from ..config import OUTPUTS_DIR, ensure_dirs
from .corpus import CorpusStore
from .linker import build_alias_map, link_text

UA = os.environ.get("QTS_NEWS_UA", "qts-research news collector research@example.com")
BROAD_FEEDS = [
    ("google-news-markets",
     "https://news.google.com/rss/search?q=stock%20market&hl=en-US&gl=US&ceid=US:en"),
    ("google-news-earnings",
     "https://news.google.com/rss/search?q=earnings%20report&hl=en-US&gl=US&ceid=US:en"),
    ("yahoo-finance", "https://finance.yahoo.com/news/rssindex"),
    ("marketwatch-top", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("prnewswire-financial",
     "https://www.prnewswire.com/rss/financial-services-latest-news/"
     "financial-services-latest-news-list.rss"),
    ("globenewswire-public",
     "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/"
     "GlobeNewswire%20-%20News%20about%20Public%20Companies"),
]
YAHOO_TICKER = ("https://feeds.finance.yahoo.com/rss/2.0/headline"
                "?s={sym}&region=US&lang=en-US")
GOOGLE_TICKER = ("https://news.google.com/rss/search?q=%22{sym}%22%20stock"
                 "&hl=en-US&gl=US&ceid=US:en")
_TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
             "guccounter", "ref", "cmpid"}
BODY_MAX_CHARS = 20_000


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    q = [(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in _TRACKING]
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path,
                       urlencode(q), ""))


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def _title_hash(title: str) -> str:
    return _sha1(re.sub(r"[^a-z0-9 ]", "", (title or "").lower()).strip())


def _fetch_body(url: str) -> tuple[str | None, bool]:
    """Main-article text via trafilatura; (None, False) on any failure."""
    try:
        import trafilatura
    except ImportError:
        return None, False
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        txt = trafilatura.extract(resp.text) if resp.text else None
        return (txt[:BODY_MAX_CHARS], True) if txt else (None, False)
    except Exception:  # noqa: BLE001
        return None, False


def _load_symbols() -> tuple[set[str], dict[str, str]]:
    """Universe symbols + alias map: try the main DB read-only, fall back to the
    cached snapshot (a pipeline writer may hold the exclusive lock right now)."""
    cache = OUTPUTS_DIR / "universe_symbols.json"
    try:
        from ..pit.store import PITStore
        from datetime import date
        with PITStore(read_only=True) as s:
            members = s.universe_as_of(date.today())
        payload = {"symbols": list(members["symbol"]),
                   "names": dict(zip(members["symbol"], members["name"].fillna("")))}
        ensure_dirs()
        cache.write_text(json.dumps(payload))
    except Exception:  # noqa: BLE001  locked or missing -> cached copy
        if not cache.exists():
            return set(), {}
        payload = json.loads(cache.read_text())
    members = pd.DataFrame(
        [{"symbol": s2, "name": n} for s2, n in payload["names"].items()])
    return set(payload["symbols"]), build_alias_map(members)


def _update_status(**kv) -> None:
    p = OUTPUTS_DIR / "corpus_status.json"
    cur = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            cur = {}
    cur.update(kv)
    p.write_text(json.dumps(cur, indent=2))


def collect_news(
    corpus: CorpusStore, symbols: set[str], alias_map: dict[str, str],
    parse=None, fetch_body=_fetch_body, max_articles: int = 80,
    batch: int = 40, state_path: Path | None = None,
) -> dict:
    if parse is None:
        import feedparser
        parse = feedparser.parse
    state_path = state_path or (OUTPUTS_DIR / "news_rotation.json")

    feeds: list[tuple[str, str, str | None]] = [(s, u, None) for s, u in BROAD_FEEDS]
    ordered = sorted(symbols)
    if batch and ordered:
        state = {}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
            except Exception:  # noqa: BLE001
                state = {}
        n = max(len(ordered), 1)
        offset = int(state.get("offset", 0)) % n
        g_offset = int(state.get("g_offset", 0)) % n
        feeds += [(f"yahoo-{s}", YAHOO_TICKER.format(sym=s), s)
                  for s in (ordered + ordered)[offset:offset + batch]]
        feeds += [(f"gnews-{s}", GOOGLE_TICKER.format(sym=s), s)
                  for s in (ordered + ordered)[g_offset:g_offset + batch]]
        state_path.write_text(json.dumps(
            {"offset": offset + batch, "g_offset": g_offset + batch}))

    rows, fetched_bodies, seen_hashes = [], 0, set()
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    for source, url, seed_sym in feeds:
        try:
            feed = parse(url)
            entries = list(getattr(feed, "entries", []))[:50]
        except Exception as e:  # noqa: BLE001
            print(f"  ! feed {source}: {e}")
            continue
        seed = {seed_sym} if seed_sym else set()
        for e in entries:
            link = canonical_url(getattr(e, "link", "") or "")
            if not link:
                continue
            h = _sha1(link)
            if h in seen_hashes:
                continue
            # Known-hash check BEFORE the body fetch: skip articles the corpus
            # already has so a rerun never re-fetches a body it already has.
            if corpus.existing_hashes("news_items", "url_hash", [h]):
                seen_hashes.add(h)
                continue
            seen_hashes.add(h)
            title = getattr(e, "title", "") or ""
            summary = re.sub(r"<[^>]+>", " ", getattr(e, "summary", "") or "")[:2000]
            body, ok = (None, False)
            # Google News links never resolve to publisher pages via plain HTTP
            # (verified 2026-07-10) — don't burn body budget on them.
            if fetched_bodies < max_articles and "news.google.com" not in link:
                body, ok = fetch_body(link)
                fetched_bodies += 1
                time.sleep(0.0 if fetch_body is not _fetch_body else 0.5)
            text_for_link = " ".join(x for x in (title, summary, body or "") if x)
            syms = seed | link_text(text_for_link, symbols, alias_map)
            pub = getattr(e, "published_parsed", None)
            claimed_published_at = None
            if pub:
                try:
                    claimed_published_at = pd.Timestamp(
                        time.strftime("%Y-%m-%d %H:%M:%S", pub))
                except Exception:  # noqa: BLE001  malformed published_parsed
                    claimed_published_at = None
            rows.append({
                "url_hash": h, "url": link, "source": source, "title": title,
                "summary": summary, "body": body,
                "claimed_published_at": claimed_published_at,
                "fetched_at": now, "symbols": ",".join(sorted(syms)),
                "title_hash": _title_hash(title), "extract_ok": ok,
            })
    if not rows:
        _update_status(last_news_run=str(now), **corpus.counts())
        return {"new_items": 0}
    df = pd.DataFrame(rows)
    n = corpus.upsert_news(df)
    _update_status(last_news_run=str(now), **corpus.counts())
    print(f"news: {n} new items ({fetched_bodies} bodies fetched)")
    return {"new_items": n}


def main() -> None:
    ensure_dirs()
    symbols, alias_map = _load_symbols()
    with CorpusStore() as corpus:
        collect_news(corpus, symbols, alias_map)
        try:
            from .deepdive import find_spikes, run_deepdive
            spikes = find_spikes(corpus)
            if spikes:
                run_deepdive(corpus, symbols, alias_map, spikes)
        except Exception as e:  # noqa: BLE001  deep-dive never blocks collection
            print(f"  ! deepdive failed: {e}")


if __name__ == "__main__":
    main()
