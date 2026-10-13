"""Crypto corpus collection: crypto subreddits + crypto news RSS -> the shared corpus,
tagged with coin symbols via the CRYPTO linker. The existing gpu scorer and
`altdata.aggregate` then flow it into alt_signals with no further work.

Reuses the equity collectors' low-level helpers (corpus upserts, body fetch, hashing) but
NOT their linker -- crypto must link cashtag/alias-only (see linker.py). Politeness and PIT
match the equity collectors: knowledge_time = fetched_at (crawl time), dedup by id/hash,
one pass per run.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pandas as pd

from ...altdata.corpus import CorpusStore
from ...altdata.news import _fetch_body, _sha1, _title_hash, canonical_url
from ...altdata.reddit import make_reddit
from ...config import ensure_dirs
from .linker import link_symbols, link_text

# Highest-signal first: the book trades BTC and ETH only, so BTC/ETH-focused trading subs
# (BitcoinMarkets, ethfinance, ethtrader) carry the most decision-relevant chatter, while the
# broad subs (CryptoCurrency, CryptoMarkets) supply the attention VOLUME that signal 3 keys on.
# SatoshiStreetBets is kept for early-attention / event surges, accepting its meme noise --
# the precision-first linker filters what actually matters.
CRYPTO_SUBREDDITS = ("CryptoCurrency", "Bitcoin", "BitcoinMarkets",
                     "ethereum", "ethtrader", "ethfinance",
                     "CryptoMarkets", "SatoshiStreetBets")

# Crypto-native news RSS. Dedicated outlets give depth and SPEED on the discrete events
# signal 3 exists to catch (hacks, depegs, ETF/regulatory); Google-News queries add breadth.
CRYPTO_NEWS_FEEDS = [
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("decrypt", "https://decrypt.co/feed"),
    ("theblock", "https://www.theblock.co/rss.xml"),
    ("bitcoinmagazine", "https://bitcoinmagazine.com/feed"),
    ("cryptoslate", "https://cryptoslate.com/feed/"),
    ("gnews-bitcoin",
     "https://news.google.com/rss/search?q=bitcoin%20price&hl=en-US&gl=US&ceid=US:en"),
    ("gnews-ethereum",
     "https://news.google.com/rss/search?q=ethereum%20price&hl=en-US&gl=US&ceid=US:en"),
    ("gnews-crypto",
     "https://news.google.com/rss/search?q=cryptocurrency%20market&hl=en-US&gl=US&ceid=US:en"),
]


def collect_reddit_crypto(reddit, corpus: CorpusStore,
                          subreddits=CRYPTO_SUBREDDITS, limit: int = 100) -> dict:
    """Fetch new posts from crypto subs, link coins, upsert. Mirrors the equity collector but
    links with the crypto linker (cashtag/alias-only)."""
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    rows: dict[str, dict] = {}
    for sub in subreddits:
        try:
            for p in reddit.subreddit(sub).new(limit=limit):
                if p.id in rows:
                    continue
                syms = link_symbols(f"{p.title}\n{p.selftext or ''}")
                author = getattr(getattr(p, "author", None), "name", "") or ""
                rows[p.id] = {
                    "post_id": p.id, "platform": "reddit",
                    "subreddit": p.subreddit.display_name,
                    "author_hash": hashlib.sha1(author.encode()).hexdigest(),
                    "title": p.title, "body": (p.selftext or "")[:10_000],
                    "score": int(p.score), "num_comments": int(p.num_comments),
                    "created_at": pd.Timestamp(p.created_utc, unit="s"),
                    "fetched_at": now, "symbols": syms,
                }
        except Exception as e:  # noqa: BLE001  one bad sub never kills the run
            print(f"  ! crypto subreddit {sub}: {e}")
    if not rows:
        return {"new_posts": 0}
    df = pd.DataFrame(list(rows.values()))
    known = corpus.existing_hashes("social_posts", "post_id", list(df["post_id"]))
    df = df[~df["post_id"].isin(known)]
    n = corpus.upsert_posts(df)
    print(f"crypto reddit: {n} new posts")
    return {"new_posts": n}


def collect_news_crypto(corpus: CorpusStore, parse=None, fetch_body=_fetch_body,
                        feeds=CRYPTO_NEWS_FEEDS, max_bodies: int = 20) -> dict:
    """Fetch crypto news RSS, link coins on title+summary, upsert. Bodies are fetched for a
    capped number of linked articles (politeness), same as the equity collector."""
    if parse is None:
        import feedparser
        parse = feedparser.parse
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    rows, bodies, seen = [], 0, set()
    for source, url in feeds:
        try:
            feed = parse(url)
        except Exception as e:  # noqa: BLE001
            print(f"  ! feed {source}: {e}")
            continue
        for e in getattr(feed, "entries", [])[:60]:
            link = canonical_url(getattr(e, "link", "") or "")
            if not link:
                continue
            uh = _sha1(link)
            if uh in seen:
                continue
            seen.add(uh)
            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or ""
            syms = link_symbols(f"{title}\n{summary}")
            body = None
            if syms and bodies < max_bodies:               # only fetch bodies for linked items
                body, ok = fetch_body(link)
                bodies += 1
            rows.append({
                "url_hash": uh, "url": link, "source": source, "title": title,
                "summary": summary[:2000], "body": body,
                "claimed_published_at": None, "fetched_at": now,
                "symbols": syms, "title_hash": _title_hash(title),
                "extract_ok": bool(body),
            })
    if not rows:
        return {"new_news": 0}
    df = pd.DataFrame(rows)
    known = corpus.existing_hashes("news_items", "url_hash", list(df["url_hash"]))
    df = df[~df["url_hash"].isin(known)]
    n = corpus.upsert_news(df)
    print(f"crypto news: {n} new items")
    return {"new_news": n}


def collect_reddit_crypto_browser(corpus: CorpusStore, subreddits=CRYPTO_SUBREDDITS,
                                  fetch=None, fetch_c=None) -> dict:
    """Crypto Reddit via the shared browser engine (posts + selective comments), linked with
    the crypto linker. This is the same engine the equity book uses -- it works without PRAW
    creds and reaches the COMMENTS, where most crypto sentiment actually lives (the post is
    often a meme image; the ticker chatter is in the thread)."""
    from ...altdata.reddit_browser import fetch_comments, fetch_listings, run_browser_engine
    fetch = fetch or (lambda: fetch_listings(subreddits))
    fetch_c = fetch_c or fetch_comments
    # symbols/alias_map are unused because link_fn is injected; pass empties.
    return run_browser_engine(corpus, set(), {}, fetch=fetch, fetch_c=fetch_c,
                              link_fn=link_text)


def main() -> None:
    ensure_dirs()
    with CorpusStore() as corpus:
        # Prefer PRAW if creds exist; otherwise the browser engine (the equity path), which
        # also gets us the comment stream. One reddit failure never blocks news.
        reddit = make_reddit()
        try:
            if reddit is not None:
                collect_reddit_crypto(reddit, corpus)
            else:
                collect_reddit_crypto_browser(corpus)
        except Exception as e:  # noqa: BLE001
            print(f"  ! crypto reddit failed: {e}")
        collect_news_crypto(corpus)


if __name__ == "__main__":
    main()
