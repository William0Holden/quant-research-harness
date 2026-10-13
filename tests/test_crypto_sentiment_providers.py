"""Store-backed provider tests. Pins the date/Timestamp-index bug that a pure-function test
could not catch, and confirms the provider reads alt_signals into the pure signals correctly.
Uses a real temp PITStore seeded with minimal rows."""
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from qts.crypto.sentiment.providers import CryptoSentimentSignals
from qts.crypto.universe import BTC
from qts.pit.store import PITStore


@pytest.fixture
def store(tmp_path):
    s = PITStore(tmp_path / "qts.duckdb")
    yield s
    s.close()


def _counts(store, symbol, concept, start: date, values):
    kt = pd.Timestamp(datetime(2020, 1, 1))  # knowledge_time well in the past -> always visible
    rows = [{"symbol": symbol, "concept": concept, "dt": start + timedelta(days=i),
             "value": float(v), "knowledge_time": kt} for i, v in enumerate(values)]
    store.write_alt_signals(pd.DataFrame(rows))


def test_attention_spike_reads_counts_and_fires(store):
    start = date(2026, 6, 1)
    # 30 quiet days then a surge, split across news + reddit concepts
    _counts(store, BTC, "news_count", start, [4] * 30 + [4, 5, 40])
    _counts(store, BTC, "reddit_count", start, [3] * 30 + [3, 4, 30])
    sig = CryptoSentimentSignals(store)
    out = sig.attention_spike(BTC, as_of=start + timedelta(days=32))
    assert out["spike"] is True and out["attention_z"] > 2


def test_vol_forecast_lifts_with_attention(store):
    start = date(2026, 6, 1)
    _counts(store, BTC, "news_count", start, [4] * 30 + [4, 5, 40])
    sig = CryptoSentimentSignals(store)
    as_of = start + timedelta(days=32)
    lifted = sig.vol_forecast(BTC, as_of, trailing_vol=0.5)
    assert lifted > 0.5                                    # attention raised the vol forecast
    # thin history -> no opinion -> trailing vol unchanged
    assert sig.vol_forecast(BTC, date(2026, 6, 3), trailing_vol=0.5) == pytest.approx(0.5)


def test_equity_risk_appetite_change_handles_date_index(store):
    """The regression: this path built a date-indexed Series and crashed comparing to a
    Timestamp. It must now run and return a finite change (or NaN), never raise."""
    store.write_universe(pd.DataFrame([
        {"symbol": "AAPL", "cik": "1", "name": "Apple", "active_from": date(2020, 1, 1),
         "active_to": None, "delisted": False}]))
    start = date(2026, 6, 1)
    rows = []
    for i in range(10):
        d = start + timedelta(days=i)
        kt = pd.Timestamp(d)          # PIT: each day's sentiment is known THAT day
        rows += [{"symbol": "AAPL", "concept": "news_sent_sum", "dt": d, "value": float(i),
                  "knowledge_time": kt},
                 {"symbol": "AAPL", "concept": "news_sent_n", "dt": d, "value": 1.0,
                  "knowledge_time": kt}]
    store.write_alt_signals(pd.DataFrame(rows))
    sig = CryptoSentimentSignals(store)
    change = sig.equity_risk_appetite_change(start + timedelta(days=9), window=5)
    assert change == change  # not NaN: rising sentiment over the window -> a real number
    assert change > 0


def test_level_never_reads_a_future_dated_row(store):
    """Defense-in-depth: a row dated AFTER as_of must never feed the level, even if its
    knowledge_time is early. alt_signals_window gates on knowledge_time, so the level trims
    dt <= as_of itself."""
    store.write_universe(pd.DataFrame([
        {"symbol": "AAPL", "cik": "1", "name": "Apple", "active_from": date(2020, 1, 1),
         "active_to": None, "delisted": False}]))
    early_kt = pd.Timestamp(datetime(2020, 1, 1))   # "known" long ago, but DATED in the future
    store.write_alt_signals(pd.DataFrame([
        {"symbol": "AAPL", "concept": "news_sent_sum", "dt": date(2026, 6, 5), "value": 5.0,
         "knowledge_time": early_kt},
        {"symbol": "AAPL", "concept": "news_sent_n", "dt": date(2026, 6, 5), "value": 1.0,
         "knowledge_time": early_kt}]))
    sig = CryptoSentimentSignals(store)
    import numpy as np
    # as_of BEFORE the row's dt -> the row is in the future -> no level
    assert np.isnan(sig.equity_risk_appetite_level(date(2026, 6, 3)))
    # as_of ON the row's dt -> visible
    assert sig.equity_risk_appetite_level(date(2026, 6, 5)) == 5.0
