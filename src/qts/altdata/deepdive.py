"""Spike-triggered news deep-dive. When a symbol's combined reddit+news mention
count today is anomalously high vs its own trailing 13-day baseline, immediately
pull the top Bing News articles WITH full bodies (Bing RSS links resolve to real
publisher pages; Google News links do not — verified 2026-07-10). Corpus-only
writes; invoked at the end of each hourly news collection run. Caps: 5 symbols
per run, 10 articles per symbol, and a symbol is deep-dived at most once per
UTC day (tracked in corpus_status.json)."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from ..config import OUTPUTS_DIR
from .corpus import CorpusStore
from .linker import link_text
from .news import _fetch_body, _sha1, _title_hash, _update_status, canonical_url

BING_TICKER = "https://www.bing.com/news/search?q=%22{sym}%22+stock&format=rss"
Z_THRESH = 3.0
MIN_MENTIONS = 5
BASELINE_DAYS = 14
MAX_SYMBOLS = 5
TOP_ARTICLES = 10


def find_spikes(corpus: CorpusStore, today: date | None = None,
                z_thresh: float = Z_THRESH,
                min_mentions: int = MIN_MENTIONS) -> list[str]:
    """Symbols whose today-count z vs their trailing baseline crosses the bar,
    strongest first, capped at MAX_SYMBOLS. Denominator floored at 1.0 count/day
    (same rationale as features/altsignals.py)."""
    today = today or datetime.now(timezone.utc).date()
    start = today - timedelta(days=BASELINE_DAYS - 1)
    parts = [corpus.daily_symbol_counts(t, start, today)
             for t in ("news_items", "social_posts", "social_comments")]
    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        return []
    piv = df.groupby(["symbol", "dt"])["n"].sum().unstack(fill_value=0.0)
    span = [str(d.date()) for d in pd.date_range(start, today, freq="D")]
    piv = piv.reindex(columns=span, fill_value=0.0)
    today_n = piv[span[-1]]
    base = piv[span[:-1]]
    z = (today_n - base.mean(axis=1)) / base.std(axis=1).clip(lower=1.0)
    hits = z[(z >= z_thresh) & (today_n >= min_mentions)]
    return list(hits.sort_values(ascending=False).index[:MAX_SYMBOLS])


def _done_today() -> set[str]:
    p = OUTPUTS_DIR / "corpus_status.json"
    try:
        st = json.loads(p.read_text()).get("deepdive_done", {})
    except Exception:  # noqa: BLE001
        return set()
    key = str(datetime.now(timezone.utc).date())
    return set(st.get(key, []))


def _mark_done(syms: list[str]) -> None:
    key = str(datetime.now(timezone.utc).date())
    done = sorted(_done_today() | set(syms))
    _update_status(deepdive_done={key: done})   # keyed by day -> old days age out


def run_deepdive(corpus: CorpusStore, symbols: set[str], alias_map: dict[str, str],
                 spike_symbols: list[str], parse=None,
                 fetch_body=_fetch_body) -> dict:
    if parse is None:
        import feedparser
        parse = feedparser.parse
    todo = [s for s in spike_symbols if s not in _done_today()]
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    rows = []
    for sym in todo:
        try:
            feed = parse(BING_TICKER.format(sym=sym))
            entries = list(getattr(feed, "entries", []))[:TOP_ARTICLES]
        except Exception as e:  # noqa: BLE001
            print(f"  ! deepdive {sym}: {e}")
            continue
        for e in entries:
            link = canonical_url(getattr(e, "link", "") or "")
            if not link:
                continue
            h = _sha1(link)
            if corpus.existing_hashes("news_items", "url_hash", [h]):
                continue
            title = getattr(e, "title", "") or ""
            body, ok = fetch_body(link)
            text = " ".join(x for x in (title, body or "") if x)
            syms = {sym} | link_text(text, symbols, alias_map)
            rows.append({
                "url_hash": h, "url": link, "source": f"bing-deepdive-{sym}",
                "title": title, "summary": "", "body": body,
                "claimed_published_at": None, "fetched_at": now,
                "symbols": ",".join(sorted(syms)),
                "title_hash": _title_hash(title), "extract_ok": ok,
            })
    n = corpus.upsert_news(pd.DataFrame(rows)) if rows else 0
    if todo:
        _mark_done(todo)
        print(f"deepdive: {todo} -> {n} new items")
    return {"new_items": n, "symbols": todo}
