import pandas as pd
from qts.altdata.corpus import CorpusStore


def _news(url_hash, symbols="AAA", fetched="2026-07-01 12:00"):
    return {"url_hash": url_hash, "url": f"https://x.test/{url_hash}",
            "source": "test", "title": "t", "summary": "s", "body": "b",
            "claimed_published_at": pd.Timestamp("2026-07-01 10:00"),
            "fetched_at": pd.Timestamp(fetched), "symbols": symbols,
            "title_hash": url_hash, "extract_ok": True}


def test_upsert_and_dedup(tmp_path):
    with CorpusStore(tmp_path / "c.duckdb") as c:
        assert c.upsert_news(pd.DataFrame([_news("h1"), _news("h2")])) == 2
        assert c.upsert_news(pd.DataFrame([_news("h1")])) == 1   # replaces, no dup
        assert c.counts()["news_items"] == 2
        assert c.existing_hashes("news_items", "url_hash", ["h1", "h9"]) == {"h1"}


def test_daily_symbol_counts(tmp_path):
    from datetime import date
    with CorpusStore(tmp_path / "c.duckdb") as c:
        c.upsert_news(pd.DataFrame([
            _news("h1", "AAA,BBB"), _news("h2", "AAA"),
            _news("h3", "", fetched="2026-07-01 13:00")]))
        df = c.daily_symbol_counts("news_items", date(2026, 6, 30), date(2026, 7, 2))
        got = {(r.symbol, str(r.dt)): r.n for r in df.itertuples()}
        assert got[("AAA", "2026-07-01")] == 2 and got[("BBB", "2026-07-01")] == 1


def _mk_post(pid, symbols="NVDA", score=1, fetched="2026-07-09 12:00:00"):
    return {"post_id": pid, "platform": "reddit", "subreddit": "stocks",
            "author_hash": "x", "title": f"t{pid}", "body": "b", "score": score,
            "num_comments": 0, "created_at": pd.Timestamp("2026-07-09"),
            "fetched_at": pd.Timestamp(fetched), "symbols": symbols}


def test_post_obs_upsert_and_history(tmp_path):
    with CorpusStore(tmp_path / "c.duckdb") as c:
        c.upsert_posts(pd.DataFrame([_mk_post("p1")]))
        obs = pd.DataFrame([
            {"post_id": "p1", "fetched_at": pd.Timestamp("2026-07-09 12:00:00"),
             "score": 1, "num_comments": 0},
            {"post_id": "p1", "fetched_at": pd.Timestamp("2026-07-09 13:00:00"),
             "score": 50, "num_comments": 9},
        ])
        assert c.upsert_post_obs(obs) == 2
        # same (post_id, fetched_at) re-upsert replaces, not duplicates
        assert c.upsert_post_obs(obs.iloc[[0]]) == 1
        n = c.con.execute("SELECT COUNT(*) FROM social_post_obs").fetchone()[0]
        assert n == 2


def test_unscored_texts_and_sentiment_roundtrip(tmp_path):
    with CorpusStore(tmp_path / "c.duckdb") as c:
        c.upsert_posts(pd.DataFrame([_mk_post("p1"), _mk_post("p2", symbols="")]))
        c.upsert_news(pd.DataFrame([{
            "url_hash": "h1", "url": "u", "source": "s", "title": "NVDA up",
            "summary": "sum", "body": None, "claimed_published_at": None,
            "fetched_at": pd.Timestamp("2026-07-09 12:00:00"), "symbols": "NVDA",
            "title_hash": "th", "extract_ok": False}]))
        un = c.unscored_texts()
        # p2 has no linked symbols -> excluded
        assert set(un["text_id"]) == {"post:p1", "news:h1"}
        c.upsert_sentiment(pd.DataFrame([{
            "text_id": "post:p1", "value": 0.5, "model": "m",
            "scored_at": pd.Timestamp("2026-07-10 09:00:00")}]))
        assert set(c.unscored_texts()["text_id"]) == {"news:h1"}


def test_daily_symbol_sentiment(tmp_path):
    from datetime import date
    with CorpusStore(tmp_path / "c.duckdb") as c:
        c.upsert_posts(pd.DataFrame([_mk_post("p1", symbols="NVDA,AMD"),
                                     _mk_post("p2", symbols="NVDA")]))
        c.upsert_sentiment(pd.DataFrame([
            {"text_id": "post:p1", "value": 1.0, "model": "m",
             "scored_at": pd.Timestamp("2026-07-10")},
            {"text_id": "post:p2", "value": -0.5, "model": "m",
             "scored_at": pd.Timestamp("2026-07-10")}]))
        out = c.daily_symbol_sentiment("social_posts", date(2026, 7, 8), date(2026, 7, 10))
        nv = out[out["symbol"] == "NVDA"].iloc[0]
        assert nv["sent_n"] == 2 and abs(nv["sent_sum"] - 0.5) < 1e-9
        am = out[out["symbol"] == "AMD"].iloc[0]
        assert am["sent_n"] == 1 and am["sent_sum"] == 1.0


def _mk_comment(cid, pid="p1", symbols="AMD", score=5, body="AMD is cooked"):
    return {"comment_id": cid, "post_id": pid, "subreddit": "stocks",
            "author_hash": "x", "body": body, "score": score,
            "created_at": pd.Timestamp("2026-07-10"),
            "fetched_at": pd.Timestamp("2026-07-10 12:00:00"), "symbols": symbols}


def test_comments_roundtrip_unscored_and_sentiment(tmp_path):
    from datetime import date
    with CorpusStore(tmp_path / "c.duckdb") as c:
        c.upsert_comments(pd.DataFrame([_mk_comment("c1"),
                                        _mk_comment("c2", symbols="")]))
        # only symbol-linked comments are scoring candidates
        assert set(c.unscored_texts()["text_id"]) == {"comment:c1"}
        c.upsert_sentiment(pd.DataFrame([{
            "text_id": "comment:c1", "value": -0.7, "model": "m",
            "scored_at": pd.Timestamp("2026-07-10 13:00:00")}]))
        assert c.unscored_texts().empty
        out = c.daily_symbol_sentiment("social_comments",
                                       date(2026, 7, 9), date(2026, 7, 10))
        row = out[out["symbol"] == "AMD"].iloc[0]
        assert row["sent_n"] == 1 and abs(row["sent_sum"] + 0.7) < 1e-9
        assert c.counts()["social_comments"] == 2


def test_comment_fetch_state(tmp_path):
    with CorpusStore(tmp_path / "c.duckdb") as c:
        c.upsert_comment_state(pd.DataFrame([{
            "post_id": "p1", "last_num_comments": 40,
            "last_fetched_at": pd.Timestamp("2026-07-10 12:00:00")}]))
        assert c.comment_fetch_state(["p1", "zz"]) == {"p1": 40}
        assert c.comment_fetch_state([]) == {}
