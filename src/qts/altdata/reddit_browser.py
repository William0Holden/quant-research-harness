"""Browser-engine Reddit collector — Patchright headless Chromium.

Reddit 403s plain HTTP from this host (verified 2026-07-10) but serves its public
listing JSON to a real browser, so we navigate to reddit once and issue in-page
fetch() calls for each subreddit's new/hot listings (the in-page-fetch pattern
proven in tiktok-mcp). No login, no API creds. Author names stored sha1-hashed,
same as the praw path. PIT: knowledge_time basis = fetched_at. Every sighting of
a post also lands a (post_id, fetched_at, score, num_comments) row in
social_post_obs so score/comment VELOCITY is computable, not just presence."""
from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timezone

import pandas as pd

from ..config import ensure_dirs
from .corpus import CorpusStore
from .linker import link_text
from .news import _load_symbols, _update_status
from .reddit import SUBREDDITS

LISTING_LIMITS = {"new": 100, "hot": 50}
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def fetch_listings(subreddits: tuple[str, ...] = SUBREDDITS) -> list[dict]:
    """One headless browser per run; returns raw reddit post dicts ('data' children)."""
    from patchright.sync_api import sync_playwright
    posts: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(user_agent=UA)
        page.goto("https://www.reddit.com/", wait_until="domcontentloaded",
                  timeout=45_000)
        for sub in subreddits:
            for kind, limit in LISTING_LIMITS.items():
                url = (f"https://www.reddit.com/r/{sub}/{kind}.json"
                       f"?limit={limit}&raw_json=1")
                try:
                    raw = page.evaluate(
                        "url => fetch(url, {credentials: 'include'})"
                        ".then(r => r.text())", url)
                    posts += [c["data"] for c in json.loads(raw)["data"]["children"]]
                except Exception as e:  # noqa: BLE001  one bad listing never kills the run
                    print(f"  ! r/{sub}/{kind}: {e}")
                time.sleep(random.uniform(1.0, 3.0))
        browser.close()
    return posts


def collect_posts_browser(corpus: CorpusStore, symbols: set[str],
                          alias_map: dict[str, str], fetch=fetch_listings,
                          link_fn=None) -> dict:
    # link_fn(text) -> set[str] of linked symbols. Defaults to the equity linker so the
    # production equity path is unchanged; the crypto collector injects its own (cashtag/
    # alias-only) linker to reuse this whole browser engine.
    if link_fn is None:
        link_fn = lambda t: link_text(t, symbols, alias_map)  # noqa: E731
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    rows: dict[str, dict] = {}
    for p in fetch():
        pid = p.get("id")
        if not pid or pid in rows:
            continue
        text = f"{p.get('title') or ''}\n{p.get('selftext') or ''}"
        syms = link_fn(text)
        rows[pid] = {
            "post_id": pid, "platform": "reddit",
            "subreddit": p.get("subreddit") or "",
            "author_hash": hashlib.sha1((p.get("author") or "").encode()).hexdigest(),
            "title": p.get("title") or "",
            "body": (p.get("selftext") or "")[:10_000],
            "score": int(p.get("score") or 0),
            "num_comments": int(p.get("num_comments") or 0),
            "created_at": pd.Timestamp(p.get("created_utc") or 0, unit="s"),
            "fetched_at": now, "symbols": ",".join(sorted(syms)),
            "stickied": bool(p.get("stickied")),   # sweep-only; not stored in DB
        }
    if not rows:
        _update_status(last_reddit_run=str(now), last_reddit_engine="browser",
                       **corpus.counts())
        print("reddit-browser: 0 posts fetched")
        return {"new_posts": 0, "obs": 0, "sweep": pd.DataFrame(
            columns=["post_id", "num_comments", "symbols", "stickied"])}
    df = pd.DataFrame(list(rows.values()))
    n_obs = corpus.upsert_post_obs(df[["post_id", "fetched_at", "score",
                                       "num_comments"]])
    known = corpus.existing_hashes("social_posts", "post_id", list(df["post_id"]))
    n = corpus.upsert_posts(df.drop(columns=["stickied"])[~df["post_id"].isin(known)])
    _update_status(last_reddit_run=str(now), last_reddit_engine="browser",
                   **corpus.counts())
    print(f"reddit-browser: {n} new posts, {n_obs} score observations")
    return {"new_posts": n, "obs": n_obs,
            "sweep": df[["post_id", "num_comments", "symbols", "stickied"]]}


# ---- comment collection -----------------------------------------------------
# The post is often a meme image with an empty body (WSB especially) and the
# daily-discussion megathreads carry most ticker chatter — the sentiment lives
# in the comments. Fetching every thread every sweep would be burst behavior,
# so selection is strict: stickied threads always, then symbol-linked, then the
# busiest, cap COMMENT_FETCHES_PER_RUN; a thread is re-fetched only after its
# comment count grows REFETCH_GROWTH-fold (tracked in comment_fetch_state).

COMMENT_FETCHES_PER_RUN = 15
TOP_COMMENTS_PER_POST = 50
MIN_COMMENT_CHARS = 12
MIN_COMMENTS_TO_FETCH = 5
REFETCH_GROWTH = 1.5
_BOT_AUTHORS = {"AutoModerator", "VisualMod"}


def select_comment_targets(corpus: CorpusStore, sweep: pd.DataFrame,
                           cap: int = COMMENT_FETCHES_PER_RUN) -> list[str]:
    """Rank this sweep's posts for comment fetching: stickied > symbol-linked >
    busiest; drop quiet threads and ones whose count hasn't grown enough."""
    if sweep.empty:
        return []
    state = corpus.comment_fetch_state(list(sweep["post_id"]))
    cand = []
    for r in sweep.itertuples():
        if r.num_comments < MIN_COMMENTS_TO_FETCH:
            continue
        last = state.get(r.post_id)
        if last is not None and r.num_comments < last * REFETCH_GROWTH:
            continue
        cand.append((not bool(r.stickied), not bool(r.symbols),
                     -int(r.num_comments), r.post_id))
    cand.sort()
    return [pid for *_, pid in cand[:cap]]


def _flatten_comments(children: list, out: list) -> None:
    for node in children or []:
        if node.get("kind") != "t1":
            continue
        d = node.get("data") or {}
        out.append(d)
        rep = d.get("replies")
        if isinstance(rep, dict):
            _flatten_comments((rep.get("data") or {}).get("children"), out)


def _usable(c: dict) -> bool:
    return (len(c.get("body") or "") >= MIN_COMMENT_CHARS
            and (c.get("author") or "") not in _BOT_AUTHORS)


def _parse_comment_listing(listing, top_n: int = TOP_COMMENTS_PER_POST) -> list[dict]:
    """[postListing, commentListing] -> top_n usable comment dicts by score."""
    if not isinstance(listing, list) or len(listing) < 2:
        return []
    flat: list[dict] = []
    _flatten_comments((listing[1].get("data") or {}).get("children"), flat)
    ok = [c for c in flat if _usable(c)]
    ok.sort(key=lambda c: -(c.get("score") or 0))
    return ok[:top_n]


def fetch_comments(post_ids: list[str]) -> dict[str, list[dict]]:
    """One headless browser; post_id -> top parsed comments."""
    from patchright.sync_api import sync_playwright
    out: dict[str, list[dict]] = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(user_agent=UA)
        page.goto("https://www.reddit.com/", wait_until="domcontentloaded",
                  timeout=45_000)
        for pid in post_ids:
            url = (f"https://www.reddit.com/comments/{pid}.json"
                   f"?limit=100&sort=top&raw_json=1")
            try:
                raw = page.evaluate(
                    "url => fetch(url, {credentials: 'include'})"
                    ".then(r => r.text())", url)
                out[pid] = _parse_comment_listing(json.loads(raw))
            except Exception as e:  # noqa: BLE001  one bad thread never kills the run
                print(f"  ! comments {pid}: {e}")
            time.sleep(random.uniform(1.0, 3.0))
        browser.close()
    return out


def collect_comments(corpus: CorpusStore, symbols: set[str],
                     alias_map: dict[str, str], raw: dict[str, list[dict]],
                     sweep_counts: dict[str, int], link_fn=None) -> dict:
    """Store new comments (symbol-linked per comment) + mark threads fetched."""
    if link_fn is None:
        link_fn = lambda t: link_text(t, symbols, alias_map)  # noqa: E731
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    rows = []
    for pid, comments in raw.items():
        for c in comments:
            cid = c.get("id")
            if not cid or not _usable(c):
                continue
            body = (c.get("body") or "")[:10_000]
            rows.append({
                "comment_id": cid, "post_id": pid,
                "subreddit": c.get("subreddit") or "",
                "author_hash": hashlib.sha1(
                    (c.get("author") or "").encode()).hexdigest(),
                "body": body, "score": int(c.get("score") or 0),
                "created_at": pd.Timestamp(c.get("created_utc") or 0, unit="s"),
                "fetched_at": now,
                "symbols": ",".join(sorted(link_fn(body))),
            })
    n = 0
    if rows:
        df = pd.DataFrame(rows).drop_duplicates("comment_id")
        known = corpus.existing_hashes("social_comments", "comment_id",
                                       list(df["comment_id"]))
        n = corpus.upsert_comments(df[~df["comment_id"].isin(known)])
    if raw:
        corpus.upsert_comment_state(pd.DataFrame(
            [{"post_id": pid, "last_num_comments": int(sweep_counts.get(pid, 0)),
              "last_fetched_at": now} for pid in raw]))
    print(f"reddit-comments: {n} new comments from {len(raw)} threads")
    return {"new_comments": n, "threads": len(raw)}


def run_browser_engine(corpus: CorpusStore, symbols: set[str],
                       alias_map: dict[str, str], fetch=fetch_listings,
                       fetch_c=fetch_comments, link_fn=None) -> dict:
    """Full browser sweep: posts + selective comment collection. This is the
    production entry point (called from reddit.main's engine switch).

    link_fn(text) -> set[str] overrides the symbol linker (default = equity). The crypto
    collector passes its cashtag/alias-only linker + crypto subreddits (via `fetch`) to reuse
    this entire engine, including the comment collection where most sentiment actually lives."""
    result = collect_posts_browser(corpus, symbols, alias_map, fetch=fetch, link_fn=link_fn)
    sweep = result.get("sweep")
    if sweep is None or sweep.empty:
        return result
    targets = select_comment_targets(corpus, sweep)
    if not targets:
        return result
    try:
        raw = fetch_c(targets)
        counts = dict(zip(sweep["post_id"], sweep["num_comments"]))
        result.update(collect_comments(corpus, symbols, alias_map, raw, counts,
                                       link_fn=link_fn))
    except Exception as e:  # noqa: BLE001  comments never block post collection
        print(f"  ! comment collection failed: {e}")
    return result


def main() -> None:
    ensure_dirs()
    symbols, alias_map = _load_symbols()
    with CorpusStore() as corpus:
        run_browser_engine(corpus, symbols, alias_map)


if __name__ == "__main__":
    main()
