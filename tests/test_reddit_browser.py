import time

import pandas as pd

from qts.altdata.corpus import CorpusStore
from qts.altdata.reddit_browser import collect_posts_browser


def _raw(pid, title="NVDA to the moon", score=10, n_comments=2, author="bob"):
    return {"id": pid, "subreddit": "wallstreetbets", "title": title,
            "selftext": "body text", "score": score, "num_comments": n_comments,
            "created_utc": 1780000000, "author": author}


def test_collect_posts_browser_upserts_posts_and_obs(tmp_path):
    with CorpusStore(tmp_path / "c.duckdb") as c:
        fetch = lambda: [_raw("a1"), _raw("a1"), _raw("b2", title="hello AMD")]
        out = collect_posts_browser(c, {"NVDA", "AMD"}, {}, fetch=fetch)
        assert out["new_posts"] == 2 and out["obs"] == 2
        posts = c.con.execute(
            "SELECT post_id, symbols, author_hash FROM social_posts ORDER BY post_id").df()
        assert list(posts["post_id"]) == ["a1", "b2"]
        assert posts.iloc[0]["symbols"] == "NVDA"
        assert posts.iloc[0]["author_hash"] != "bob"  # hashed, never raw

        # second sweep: same posts, new scores -> 0 new posts, 2 new observations
        time.sleep(0.01)  # distinct fetched_at for the (post_id, fetched_at) PK
        fetch2 = lambda: [_raw("a1", score=999), _raw("b2")]
        out2 = collect_posts_browser(c, {"NVDA", "AMD"}, {}, fetch=fetch2)
        assert out2["new_posts"] == 0 and out2["obs"] == 2
        hist = c.con.execute(
            "SELECT score FROM social_post_obs WHERE post_id='a1' "
            "ORDER BY fetched_at").df()
        assert list(hist["score"]) == [10, 999]


def test_collect_posts_browser_empty_fetch_is_noop(tmp_path):
    with CorpusStore(tmp_path / "c.duckdb") as c:
        out = collect_posts_browser(c, {"NVDA"}, {}, fetch=lambda: [])
        assert out["new_posts"] == 0 and out["obs"] == 0
        assert out["sweep"].empty


def _sweep_row(pid, n_comments, symbols="", stickied=False):
    return {"post_id": pid, "num_comments": n_comments, "symbols": symbols,
            "stickied": stickied}


def test_select_comment_targets_priority_growth_and_cap(tmp_path):
    from qts.altdata.reddit_browser import select_comment_targets
    with CorpusStore(tmp_path / "c.duckdb") as c:
        # previously fetched: p_stale at 30 comments, p_grown at 30 comments
        c.upsert_comment_state(pd.DataFrame([
            {"post_id": "p_stale", "last_num_comments": 30,
             "last_fetched_at": pd.Timestamp("2026-07-10 10:00:00")},
            {"post_id": "p_grown", "last_num_comments": 30,
             "last_fetched_at": pd.Timestamp("2026-07-10 10:00:00")}]))
        sweep = pd.DataFrame([
            _sweep_row("p_daily", 5000, stickied=True),      # megathread: always
            _sweep_row("p_linked", 20, symbols="NVDA"),      # linked: high prio
            _sweep_row("p_busy", 300),                       # busiest unlinked
            _sweep_row("p_quiet", 3),                        # < 5 comments: never
            _sweep_row("p_stale", 40),                       # 40 < 30*1.5: skip
            _sweep_row("p_grown", 100),                      # 100 >= 45: refetch
        ])
        got = select_comment_targets(c, sweep)
        assert got[0] == "p_daily"                # stickied first
        assert set(got) == {"p_daily", "p_linked", "p_busy", "p_grown"}
        assert select_comment_targets(c, sweep, cap=2) == ["p_daily", "p_linked"]


def test_collect_comments_filters_links_and_updates_state(tmp_path):
    from qts.altdata.reddit_browser import collect_comments
    raw = {"p1": [
        {"id": "c1", "body": "NVDA earnings will crush it this quarter",
         "score": 50, "author": "alice", "subreddit": "stocks",
         "created_utc": 1780000000},
        {"id": "c2", "body": "lol", "score": 900, "author": "bob",
         "subreddit": "stocks", "created_utc": 1780000000},   # too short: dropped
        {"id": "c3", "body": "I am a bot, beep boop, read the rules please",
         "score": 1, "author": "AutoModerator", "subreddit": "stocks",
         "created_utc": 1780000000},                          # automod: dropped
    ]}
    with CorpusStore(tmp_path / "c.duckdb") as c:
        out = collect_comments(c, {"NVDA"}, {}, raw,
                               sweep_counts={"p1": 120})
        assert out["new_comments"] == 1
        row = c.con.execute(
            "SELECT comment_id, post_id, symbols, author_hash "
            "FROM social_comments").df().iloc[0]
        assert row["comment_id"] == "c1" and row["post_id"] == "p1"
        assert row["symbols"] == "NVDA" and row["author_hash"] != "alice"
        assert c.comment_fetch_state(["p1"]) == {"p1": 120}
        # re-run: same comments -> 0 new
        out2 = collect_comments(c, {"NVDA"}, {}, raw, sweep_counts={"p1": 120})
        assert out2["new_comments"] == 0


def test_flatten_comment_tree_and_top_cap():
    from qts.altdata.reddit_browser import _parse_comment_listing
    def node(cid, score, replies=None):
        d = {"id": cid, "body": f"body of {cid} long enough", "score": score,
             "author": "u", "subreddit": "s", "created_utc": 1}
        if replies:
            d["replies"] = {"kind": "Listing",
                            "data": {"children": replies}}
        return {"kind": "t1", "data": d}
    listing = [{"kind": "Listing", "data": {"children": []}},   # post listing
               {"kind": "Listing", "data": {"children": [
                   node("a", 5, replies=[node("a1", 99)]),
                   node("b", 10),
                   {"kind": "more", "data": {}},
               ]}}]
    got = _parse_comment_listing(listing, top_n=2)
    assert [c["id"] for c in got] == ["a1", "b"]   # nested reply found, top-2 by score


def test_run_browser_engine_runs_both_phases(tmp_path):
    """Regression: the cron entry point (reddit.main engine switch) must reach
    the comment phase — collect_posts_browser alone does not."""
    from qts.altdata.reddit_browser import run_browser_engine
    raw_post = _raw("p1", title="NVDA thread")
    raw_post["num_comments"] = 50
    fetch_c_calls = []

    def fetch_c(targets):
        fetch_c_calls.append(targets)
        return {t: [{"id": f"cm_{t}", "body": "NVDA going up for sure",
                     "score": 3, "author": "u", "subreddit": "s",
                     "created_utc": 1}] for t in targets}

    with CorpusStore(tmp_path / "c.duckdb") as c:
        out = run_browser_engine(c, {"NVDA"}, {}, fetch=lambda: [raw_post],
                                 fetch_c=fetch_c)
        assert fetch_c_calls == [["p1"]]
        assert out["new_comments"] == 1
        assert c.comment_fetch_state(["p1"]) == {"p1": 50}
