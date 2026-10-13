"""Reddit collector via the official API (praw). Read-only OAuth app creds in env:
REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT. Without creds the CLI
prints one line and exits 0 (collector is a no-op until the user registers the app).
Authors are stored as sha1 hashes — we analyze aggregates, not people."""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

import pandas as pd

from ..config import ensure_dirs
from .corpus import CorpusStore
from .linker import link_text
from .news import _load_symbols, _update_status

SUBREDDITS = ("wallstreetbets", "stocks", "investing",
              "StockMarket", "options", "SecurityAnalysis")


def make_reddit():
    cid = os.environ.get("REDDIT_CLIENT_ID")
    csec = os.environ.get("REDDIT_CLIENT_SECRET")
    if not (cid and csec):
        return None
    import praw
    return praw.Reddit(
        client_id=cid, client_secret=csec,
        user_agent=os.environ.get(
            "REDDIT_USER_AGENT", "qts-research:collector:v1 (by /u/qts_research)"))


def collect_posts(reddit, corpus: CorpusStore, symbols: set[str],
                  alias_map: dict[str, str],
                  subreddits: tuple[str, ...] = SUBREDDITS, limit: int = 100) -> dict:
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    rows: dict[str, dict] = {}
    for sub in subreddits:
        try:
            for p in reddit.subreddit(sub).new(limit=limit):
                if p.id in rows:
                    continue
                text = f"{p.title}\n{p.selftext or ''}"
                syms = link_text(text, symbols, alias_map)
                author = getattr(getattr(p, "author", None), "name", "") or ""
                rows[p.id] = {
                    "post_id": p.id, "platform": "reddit",
                    "subreddit": p.subreddit.display_name,
                    "author_hash": hashlib.sha1(author.encode()).hexdigest(),
                    "title": p.title, "body": (p.selftext or "")[:10_000],
                    "score": int(p.score), "num_comments": int(p.num_comments),
                    "created_at": pd.Timestamp(p.created_utc, unit="s"),
                    "fetched_at": now, "symbols": ",".join(sorted(syms)),
                }
        except Exception as e:  # noqa: BLE001  one bad sub never kills the run
            print(f"  ! subreddit {sub}: {e}")
    if not rows:
        _update_status(last_reddit_run=str(now), **corpus.counts())
        return {"new_posts": 0}
    df = pd.DataFrame(list(rows.values()))
    known = corpus.existing_hashes("social_posts", "post_id", list(df["post_id"]))
    df = df[~df["post_id"].isin(known)]
    n = corpus.upsert_posts(df)
    _update_status(last_reddit_run=str(now), **corpus.counts())
    print(f"reddit: {n} new posts")
    return {"new_posts": n}


def main() -> None:
    ensure_dirs()
    symbols, alias_map = _load_symbols()
    reddit = make_reddit()
    with CorpusStore() as corpus:
        if reddit is not None:
            collect_posts(reddit, corpus, symbols, alias_map)
        else:
            from .reddit_browser import run_browser_engine
            run_browser_engine(corpus, symbols, alias_map)


if __name__ == "__main__":
    main()
