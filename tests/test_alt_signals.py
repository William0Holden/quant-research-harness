"""tests/test_alt_signals.py"""
import math
from datetime import date, timedelta
import pandas as pd
from qts.altdata.aggregate import run_aggregation
from qts.altdata.corpus import CorpusStore
from qts.features.altsignals import NewsVolumeAnomaly
from qts.pit.store import PITStore


def test_store_roundtrip_and_pit(tmp_path):
    s = PITStore(tmp_path / "t.duckdb")
    s.write_alt_signals(pd.DataFrame([
        {"symbol": "AAA", "concept": "news_count", "dt": date(2026, 7, 1),
         "value": 3.0, "knowledge_time": pd.Timestamp("2026-07-01 23:00")}]))
    ok = s.alt_signals_window(["AAA"], "news_count", date(2026, 6, 1),
                              pd.Timestamp("2026-07-02"))
    early = s.alt_signals_window(["AAA"], "news_count", date(2026, 6, 1),
                                 pd.Timestamp("2026-07-01 12:00"))
    assert len(ok) == 1 and len(early) == 0        # knowledge_time gate holds
    s.close()


def test_aggregation_moves_counts(tmp_path):
    corpus = CorpusStore(tmp_path / "c.duckdb")
    corpus.upsert_news(pd.DataFrame([{
        "url_hash": "h1", "url": "u", "source": "s", "title": "t", "summary": "",
        "body": None, "claimed_published_at": None,
        "fetched_at": pd.Timestamp("2026-07-01 12:00"), "symbols": "AAA",
        "title_hash": "th", "extract_ok": False}]))
    store = PITStore(tmp_path / "t.duckdb")
    out = run_aggregation(corpus, store, days_back=30,
                          today=date(2026, 7, 2))
    assert out["news_count"] == 1
    # run_aggregation stamps knowledge_time with the real clock, so as_of must be
    # relative to now, not a fixed date (a hardcoded as_of expires once the wall
    # clock passes it and the PIT gate correctly hides the row).
    as_of = pd.Timestamp.now(tz="UTC").tz_localize(None) + pd.Timedelta(days=1)
    df = store.alt_signals_window(["AAA"], "news_count", date(2026, 6, 1), as_of)
    assert df.iloc[0]["value"] == 1.0
    corpus.close(); store.close()


def test_aggregation_preserves_knowledge_time_when_unchanged(tmp_path):
    # Regression for I3: re-upserting the trailing window nightly must not restamp
    # knowledge_time=now on rows whose value didn't change -- only NEW/CHANGED rows
    # should be written.
    corpus = CorpusStore(tmp_path / "c.duckdb")
    corpus.upsert_news(pd.DataFrame([{
        "url_hash": "h1", "url": "u1", "source": "s", "title": "t", "summary": "",
        "body": None, "claimed_published_at": None,
        "fetched_at": pd.Timestamp("2026-07-01 12:00"), "symbols": "AAA",
        "title_hash": "th1", "extract_ok": False}]))
    store = PITStore(tmp_path / "t.duckdb")

    out1 = run_aggregation(corpus, store, days_back=30, today=date(2026, 7, 2))
    assert out1["news_count"] == 1
    row1 = store.con.execute(
        "SELECT knowledge_time, value FROM alt_signals "
        "WHERE symbol = 'AAA' AND concept = 'news_count'").df().iloc[0]
    assert row1["value"] == 1.0
    kt1 = row1["knowledge_time"]

    # second run on the identical corpus -> nothing changed, nothing written
    out2 = run_aggregation(corpus, store, days_back=30, today=date(2026, 7, 2))
    assert out2["news_count"] == 0
    row2 = store.con.execute(
        "SELECT knowledge_time, value FROM alt_signals "
        "WHERE symbol = 'AAA' AND concept = 'news_count'").df().iloc[0]
    assert row2["value"] == 1.0
    assert row2["knowledge_time"] == kt1              # kt untouched

    # add one more corpus item for the same (symbol, day) -> value 1 -> 2, changed
    corpus.upsert_news(pd.DataFrame([{
        "url_hash": "h2", "url": "u2", "source": "s", "title": "t2", "summary": "",
        "body": None, "claimed_published_at": None,
        "fetched_at": pd.Timestamp("2026-07-01 13:00"), "symbols": "AAA",
        "title_hash": "th2", "extract_ok": False}]))
    out3 = run_aggregation(corpus, store, days_back=30, today=date(2026, 7, 2))
    assert out3["news_count"] == 1                    # exactly the one changed row
    row3 = store.con.execute(
        "SELECT knowledge_time, value FROM alt_signals "
        "WHERE symbol = 'AAA' AND concept = 'news_count'").df().iloc[0]
    assert row3["value"] == 2.0
    assert row3["knowledge_time"] > kt1                # freshly restamped

    corpus.close(); store.close()


def _seed_counts(s, sym, base, spike_day, spike):
    rows = []
    for i in range(70):
        d = date(2026, 4, 1) + timedelta(days=i)
        v = spike if d == spike_day else base
        rows.append({"symbol": sym, "concept": "news_count", "dt": d, "value": v,
                     "knowledge_time": pd.Timestamp(d) + pd.Timedelta(hours=23)})
    s.write_alt_signals(pd.DataFrame(rows))


def test_news_anomaly_detects_spike(tmp_path):
    s = PITStore(tmp_path / "t.duckdb")
    _seed_counts(s, "HOT", base=2.0, spike_day=date(2026, 6, 8), spike=30.0)
    _seed_counts(s, "CALM", base=2.0, spike_day=date(2026, 6, 8), spike=2.0)
    v = NewsVolumeAnomaly().compute(s, ["HOT", "CALM", "NODATA"], date(2026, 6, 9))
    assert v["HOT"] > v["CALM"]
    assert pd.isna(v["NODATA"])                    # no history -> self-excludes
    s.close()


def test_anomaly_flat_base_denominator_floor(tmp_path):
    s = PITStore(tmp_path / "t.duckdb")
    # quiet symbol: perfectly flat 0.0 base (std=0) with a small recent bump.
    # Old code: z = diff / (0 + 1e-9) -> ~1e9, exploding the cross-section.
    rows = []
    for i in range(70):
        d = date(2026, 4, 1) + timedelta(days=i)
        v = 2.0 if d >= date(2026, 6, 2) else 0.0   # bump only in the recent window
        rows.append({"symbol": "QUIET", "concept": "news_count", "dt": d, "value": v,
                     "knowledge_time": pd.Timestamp(d) + pd.Timedelta(hours=23)})
    s.write_alt_signals(pd.DataFrame(rows))
    # normal-variance symbols from the existing spike-detection pattern, seeded
    # in the same store, to confirm they're unaffected by the floor/clip.
    _seed_counts(s, "HOT", base=2.0, spike_day=date(2026, 6, 8), spike=30.0)
    _seed_counts(s, "CALM", base=2.0, spike_day=date(2026, 6, 8), spike=2.0)

    v = NewsVolumeAnomaly().compute(s, ["QUIET", "HOT", "CALM"], date(2026, 6, 9))

    assert math.isfinite(v["QUIET"])
    assert abs(v["QUIET"]) <= 10.0
    assert v["HOT"] > v["CALM"]                     # unaffected in sign/magnitude
    s.close()


def test_anomaly_nan_when_history_short(tmp_path):
    s = PITStore(tmp_path / "t.duckdb")
    rows = [{"symbol": "NEW", "concept": "news_count", "dt": date(2026, 6, 1),
             "value": 5.0, "knowledge_time": pd.Timestamp("2026-06-01 23:00")}]
    s.write_alt_signals(pd.DataFrame(rows))
    v = NewsVolumeAnomaly().compute(s, ["NEW"], date(2026, 6, 9))
    assert pd.isna(v["NEW"])                       # < MIN_HISTORY_DAYS of coverage
    s.close()


def test_aggregation_writes_sentiment_concepts(tmp_path):
    from datetime import date
    from qts.altdata.aggregate import run_aggregation
    from qts.altdata.corpus import CorpusStore
    from qts.pit.store import PITStore

    with CorpusStore(tmp_path / "c.duckdb") as corpus, \
         PITStore(tmp_path / "q.duckdb") as store:
        corpus.upsert_posts(pd.DataFrame([{
            "post_id": "p1", "platform": "reddit", "subreddit": "stocks",
            "author_hash": "x", "title": "NVDA", "body": "", "score": 1,
            "num_comments": 0, "created_at": pd.Timestamp("2026-07-09"),
            "fetched_at": pd.Timestamp("2026-07-09 12:00:00"), "symbols": "NVDA"}]))
        corpus.upsert_sentiment(pd.DataFrame([{
            "text_id": "post:p1", "value": 0.8, "model": "m",
            "scored_at": pd.Timestamp("2026-07-10 09:00:00")}]))
        run_aggregation(corpus, store, today=date(2026, 7, 10))
        df = store.con.execute(
            "SELECT concept, value FROM alt_signals WHERE symbol='NVDA' "
            "AND concept LIKE 'reddit_sent%' ORDER BY concept").df()
        assert dict(zip(df["concept"], df["value"])) == {
            "reddit_sent_n": 1.0, "reddit_sent_sum": 0.8}


def test_aggregation_counts_comments_as_reddit_mentions(tmp_path):
    from datetime import date
    from qts.altdata.corpus import CorpusStore

    with CorpusStore(tmp_path / "c.duckdb") as corpus, \
         PITStore(tmp_path / "q.duckdb") as store:
        corpus.upsert_posts(pd.DataFrame([{
            "post_id": "p1", "platform": "reddit", "subreddit": "stocks",
            "author_hash": "x", "title": "NVDA", "body": "", "score": 1,
            "num_comments": 2, "created_at": pd.Timestamp("2026-07-09"),
            "fetched_at": pd.Timestamp("2026-07-09 12:00:00"), "symbols": "NVDA"}]))
        corpus.upsert_comments(pd.DataFrame([
            {"comment_id": f"c{i}", "post_id": "p1", "subreddit": "stocks",
             "author_hash": "x", "body": "NVDA talk", "score": 1,
             "created_at": pd.Timestamp("2026-07-09"),
             "fetched_at": pd.Timestamp("2026-07-09 13:00:00"), "symbols": "NVDA"}
            for i in range(2)]))
        corpus.upsert_sentiment(pd.DataFrame([
            {"text_id": "comment:c0", "value": -1.0, "model": "m",
             "scored_at": pd.Timestamp("2026-07-10")},
            {"text_id": "post:p1", "value": 1.0, "model": "m",
             "scored_at": pd.Timestamp("2026-07-10")}]))
        run_aggregation(corpus, store, today=date(2026, 7, 10))
        df = store.con.execute(
            "SELECT concept, value FROM alt_signals WHERE symbol='NVDA' "
            "AND concept IN ('reddit_count','reddit_sent_sum','reddit_sent_n') "
            "ORDER BY concept").df()
        got = dict(zip(df["concept"], df["value"]))
        assert got["reddit_count"] == 3.0       # 1 post + 2 comments
        assert got["reddit_sent_n"] == 2.0      # post +1.0, comment -1.0
        assert abs(got["reddit_sent_sum"]) < 1e-9
