from datetime import date

import pandas as pd
import pytest

from qts.ingest import prices as prices_mod
from qts.pit.store import PITStore


@pytest.fixture
def store(tmp_path):
    s = PITStore(tmp_path / "t.duckdb")
    yield s
    s.close()


def _bar(symbol, d):
    return {
        "symbol": symbol, "dt": d, "close": 100.0, "adj_close": 100.0,
        "volume": 1000, "knowledge_time": pd.Timestamp(d) + pd.Timedelta(hours=23),
    }


def test_last_dates(store):
    store.write_prices(pd.DataFrame([_bar("AAA", date(2026, 6, 20)),
                                     _bar("AAA", date(2026, 6, 23)),
                                     _bar("BBB", date(2026, 6, 10))]))
    out = prices_mod.last_dates(store, ["AAA", "BBB", "CCC"])
    assert out["AAA"] == date(2026, 6, 23)
    assert out["BBB"] == date(2026, 6, 10)
    assert "CCC" not in out  # never ingested -> absent, caller uses epoch 0


def test_ingest_prices_incremental_passes_start_epoch(store, monkeypatch):
    store.write_prices(pd.DataFrame([_bar("AAA", date(2026, 6, 23))]))
    seen = {}

    def fake_fetch(symbol, start_epoch=0):
        seen[symbol] = start_epoch
        return pd.DataFrame()

    monkeypatch.setattr(prices_mod, "fetch_prices", fake_fetch)
    prices_mod.ingest_prices(store, ["AAA", "CCC"], sleep_s=0.0, incremental=True)
    # AAA resumes from (2026-06-23 - 7d) = 2026-06-16 UTC midnight
    assert seen["AAA"] == int(pd.Timestamp("2026-06-16", tz="UTC").timestamp())
    assert seen["CCC"] == 0  # unknown symbol -> full history


def test_ingest_prices_full_by_default(store, monkeypatch):
    seen = {}

    def fake_fetch(symbol, start_epoch=0):
        seen[symbol] = start_epoch
        return pd.DataFrame()

    monkeypatch.setattr(prices_mod, "fetch_prices", fake_fetch)
    prices_mod.ingest_prices(store, ["AAA"], sleep_s=0.0)
    assert seen["AAA"] == 0


def _bars(symbol, dts, adj_close):
    return pd.DataFrame([
        {
            "symbol": symbol, "dt": d, "close": adj_close, "adj_close": adj_close,
            "volume": 1000, "knowledge_time": pd.Timestamp(d) + pd.Timedelta(hours=23),
        }
        for d in dts
    ])


def test_incremental_split_triggers_full_refetch(store, monkeypatch):
    dts = [date(2026, 6, d) for d in range(10, 20)]
    store.write_prices(_bars("AAA", dts, 100.0))

    calls = []

    def fake_fetch(symbol, start_epoch=0):
        calls.append((symbol, start_epoch))
        if symbol != "AAA":
            return pd.DataFrame()
        if start_epoch == 0:
            # Full history refetch — distinguishable value proves the full-refetch path won.
            return _bars("AAA", dts, 25.0)
        # Partial/incremental refetch: overlap rows come back with a shifted adjustment
        # basis (as if Yahoo just applied a 2:1 split retroactively across the series).
        return _bars("AAA", dts[-3:], 50.0)

    monkeypatch.setattr(prices_mod, "fetch_prices", fake_fetch)
    prices_mod.ingest_prices(store, ["AAA"], sleep_s=0.0, incremental=True)

    aaa_calls = [c for c in calls if c[0] == "AAA"]
    assert len(aaa_calls) == 2
    assert aaa_calls[0][1] > 0          # first call resumed from the incremental epoch
    assert aaa_calls[1][1] == 0         # second call is a full refetch from epoch 0

    on_disk = store.con.execute(
        "SELECT DISTINCT adj_close FROM prices WHERE symbol = 'AAA'"
    ).df()
    assert set(on_disk["adj_close"]) == {25.0}


def test_incremental_clean_overlap_no_refetch(store, monkeypatch):
    dts = [date(2026, 6, d) for d in range(10, 20)]
    store.write_prices(_bars("AAA", dts, 100.0))

    calls = []

    def fake_fetch(symbol, start_epoch=0):
        calls.append((symbol, start_epoch))
        if symbol != "AAA":
            return pd.DataFrame()
        # Overlap rows match the stored basis exactly -> no split/dividend shift.
        return _bars("AAA", dts[-3:], 100.0)

    monkeypatch.setattr(prices_mod, "fetch_prices", fake_fetch)
    prices_mod.ingest_prices(store, ["AAA"], sleep_s=0.0, incremental=True)

    aaa_calls = [c for c in calls if c[0] == "AAA"]
    assert len(aaa_calls) == 1

    on_disk = store.con.execute(
        "SELECT DISTINCT adj_close FROM prices WHERE symbol = 'AAA'"
    ).df()
    assert set(on_disk["adj_close"]) == {100.0}
