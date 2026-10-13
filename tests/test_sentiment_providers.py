from datetime import date, timedelta

import pandas as pd

from qts.config import DEFAULT_LIVE_GATED
from qts.features.sentiment import NewsSentiment, RedditSentiment
from qts.pit.store import PITStore

AS_OF = date(2026, 7, 10)


def _seed(store, symbol, rows, prefix="reddit_sent"):
    """rows: list of (days_ago, sent_sum, sent_n) relative to AS_OF."""
    recs = []
    for days_ago, s, n in rows:
        dt = AS_OF - timedelta(days=days_ago)
        recs += [{"symbol": symbol, "concept": f"{prefix}_sum", "dt": dt,
                  "value": s, "knowledge_time": pd.Timestamp(dt)},
                 {"symbol": symbol, "concept": f"{prefix}_n", "dt": dt,
                  "value": n, "knowledge_time": pd.Timestamp(dt)}]
    store.write_alt_signals(pd.DataFrame(recs))


def test_decayed_weighted_mean_and_min_mentions(tmp_path):
    with PITStore(tmp_path / "q.duckdb") as store:
        # HOT: 4 mentions today, all bullish 1.0 -> value ~1.0
        _seed(store, "HOT", [(0, 4.0, 4)])
        # OLDN: 4 bullish mentions 10 days ago, 1 bearish today ->
        # decayed weight pulls value toward today's bearish print
        _seed(store, "OLDN", [(10, 4.0, 4), (0, -1.0, 1)])
        # THIN: only 2 mentions total -> NaN (< 3 minimum)
        _seed(store, "THIN", [(0, 2.0, 2)])
        vals = RedditSentiment().compute(store, ["HOT", "OLDN", "THIN", "NONE"], AS_OF)
        assert abs(vals["HOT"] - 1.0) < 1e-9
        assert vals["OLDN"] < 0.5          # decay downweights the old bullish block
        assert pd.isna(vals["THIN"]) and pd.isna(vals["NONE"])


def test_news_provider_reads_news_concepts_and_is_gated(tmp_path):
    with PITStore(tmp_path / "q.duckdb") as store:
        _seed(store, "AAA", [(0, -3.0, 3)], prefix="news_sent")
        vals = NewsSentiment().compute(store, ["AAA"], AS_OF)
        assert abs(vals["AAA"] + 1.0) < 1e-9
    assert {"reddit_sentiment", "news_sentiment"} <= set(DEFAULT_LIVE_GATED)
