"""Daily prices from Yahoo's key-free chart API.

knowledge_time = dt (EOD): a day's close is knowable after that day's close. The backtest
fills on the NEXT bar, so there is no same-bar look-ahead.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
HEADERS = {"User-Agent": "Mozilla/5.0 (qts-research)"}


def fetch_prices(symbol: str, start_epoch: int = 0) -> pd.DataFrame:
    now = int(datetime.now(timezone.utc).timestamp())
    params = {
        "period1": start_epoch, "period2": now,
        "interval": "1d", "events": "div,splits",
    }
    r = requests.get(CHART.format(sym=symbol), params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"]
    if not res:
        return pd.DataFrame()
    res = res[0]
    ts = res.get("timestamp")
    if not ts:
        return pd.DataFrame()
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    dts = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize()
    df = pd.DataFrame({
        "symbol": symbol,
        "dt": dts.date,
        "close": q.get("close"),
        "adj_close": adj if adj is not None else q.get("close"),
        "volume": q.get("volume"),
    })
    df = df.dropna(subset=["close"]).copy()
    df["volume"] = df["volume"].fillna(0).astype("int64")
    # knowledge_time = EOD of the bar date
    df["knowledge_time"] = pd.to_datetime(df["dt"]) + pd.Timedelta(hours=23, minutes=59)
    return df


def _overlap_mismatch(store, symbol: str, df: pd.DataFrame, rtol: float = 5e-4) -> bool:
    """True when Yahoo's back-adjustment basis shifted (dividend/split) since our last
    fetch — detected by comparing refetched overlap bars to stored bars."""
    if df.empty:
        return False
    dts = pd.to_datetime(df["dt"])
    stored = store.con.execute(
        "SELECT dt, close, adj_close FROM prices WHERE symbol = ? AND dt >= ? AND dt <= ?",
        [symbol, dts.min().date(), dts.max().date()],
    ).df()
    if stored.empty:
        # No overlap rows to compare (gap between resume window and stored
        # history) — basis check SKIPPED, not passed.
        print(f"  ? {symbol}: no overlap bars to verify adjustment basis")
        return False
    stored["dt"] = pd.to_datetime(stored["dt"])
    fetched = df.assign(dt=dts)[["dt", "close", "adj_close"]]
    merged = fetched.merge(stored, on="dt", suffixes=("_new", "_old"))
    if merged.empty or len(merged) != len(stored):
        return True  # row count mismatch prevents a clean comparison -> refetch to be safe
    close_ok = np.isclose(merged["close_new"], merged["close_old"], rtol=rtol)
    adj_ok = np.isclose(merged["adj_close_new"], merged["adj_close_old"], rtol=rtol)
    return not bool(close_ok.all() and adj_ok.all())


def last_dates(store, symbols: list[str]) -> dict[str, date]:
    """Latest ingested bar date per symbol (absent if never ingested).

    Ingest bookmark only — raw MAX(dt), not knowledge_time-filtered; never use as a
    PIT accessor.
    """
    df = store.con.execute(
        "SELECT symbol, MAX(dt) AS last_dt FROM prices GROUP BY symbol"
    ).df()
    m = dict(zip(df["symbol"], df["last_dt"]))
    return {s: m[s].date() for s in symbols if s in m and pd.notna(m[s])}


def ingest_prices(
    store, symbols: list[str], sleep_s: float = 0.4, incremental: bool = False,
) -> dict[str, int]:
    """Fetch+upsert daily bars. incremental=True resumes each symbol from
    max(dt) - 7d (overlap re-upserts harmlessly on the (symbol, dt) PK)."""
    starts: dict[str, int] = {}
    if incremental:
        for sym, last in last_dates(store, symbols).items():
            resume = pd.Timestamp(last, tz="UTC") - pd.Timedelta(days=7)
            starts[sym] = int(resume.timestamp())
    out: dict[str, int] = {}
    for sym in symbols:
        try:
            df = fetch_prices(sym, start_epoch=starts.get(sym, 0))
            if incremental and sym in starts and _overlap_mismatch(store, sym, df):
                print(f"  ~ {sym}: adjustment basis changed, full refetch")
                df = fetch_prices(sym)
            out[sym] = store.write_prices(df)
        except Exception as e:  # noqa: BLE001
            out[sym] = -1
            print(f"  ! prices {sym}: {e}")
        time.sleep(sleep_s)
    return out
