"""8-K event features from the PIT events table (item codes ingested by qts.ingest.sec).

All emit 0.0 for event-less symbols (existing event-provider convention); the
ranker's price-eligibility gate excludes data-dead names. Direction LOWER_BETTER:
officer churn, non-reliance (accounting klaxon), and abnormal 8-K frequency are
historically bad news. All are LIVE-GATED (config.DEFAULT_LIVE_GATED) until IC-proven.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..pit.store import PITStore
from ..types import Backtestability, Direction, FeatureSpec
from .base import FeatureProvider, register


def _decayed_counts(ev: pd.DataFrame, symbols: list[str], as_of: date,
                    half_life_days: float = 21.0) -> pd.Series:
    out = pd.Series(0.0, index=symbols)
    if ev.empty:
        return out
    age = (pd.Timestamp(as_of) - pd.to_datetime(ev["filing_date"])).dt.days
    w = np.exp(-np.log(2.0) * age / half_life_days)
    s = pd.Series(w.values, index=ev["symbol"].values).groupby(level=0).sum()
    return s.reindex(symbols).fillna(0.0)


class OfficerChange(FeatureProvider):
    spec = FeatureSpec(
        name="event_officer_change", direction=Direction.LOWER_BETTER,
        backtestability=Backtestability.BACKTESTABLE,
        description="Half-life-decayed count of 8-K item 5.02 (officer/director change), 90d window.")

    def compute(self, store: PITStore, symbols: list[str], as_of: date) -> pd.Series:
        ev = store.events_between(symbols, "8-K", as_of - timedelta(days=90), as_of)
        sel = ev[ev["item"].astype(str).str.startswith("5.02")]
        return _decayed_counts(sel, symbols, as_of)


class NonReliance(FeatureProvider):
    spec = FeatureSpec(
        name="event_non_reliance", direction=Direction.LOWER_BETTER,
        backtestability=Backtestability.BACKTESTABLE,
        description="1.0 if an 8-K item 4.02 (non-reliance on prior financials) filed in past 365d.")

    def compute(self, store: PITStore, symbols: list[str], as_of: date) -> pd.Series:
        ev = store.events_between(symbols, "8-K", as_of - timedelta(days=365), as_of)
        flagged = set(ev[ev["item"].astype(str).str.startswith("4.02")]["symbol"])
        return pd.Series([1.0 if s in flagged else 0.0 for s in symbols], index=symbols)


class EightKFreqAnomaly(FeatureProvider):
    spec = FeatureSpec(
        name="eightk_freq_anomaly", direction=Direction.LOWER_BETTER,
        backtestability=Backtestability.BACKTESTABLE,
        description="log1p(8-K count last 90d) - log1p(90d-scaled count of prior 365d).")

    def compute(self, store: PITStore, symbols: list[str], as_of: date) -> pd.Series:
        # Single fetch: events_between has no upper bound on filing_date for `start`
        # (it only bounds via knowledge_time <= as_of), so a second call scoped to the
        # base window would still pull recent-window rows. Fetch the full 455d span
        # once and split recent/base by filing_date in pandas instead.
        cutoff = pd.Timestamp(as_of - timedelta(days=90))
        ev = store.events_between(symbols, "8-K", as_of - timedelta(days=455), as_of)
        fdate = pd.to_datetime(ev["filing_date"])
        recent = ev[fdate > cutoff]
        base = ev[fdate <= cutoff]

        # count unique filings (one row per item code -> dedup by accession)
        rc = recent.drop_duplicates("accession").groupby("symbol").size()
        bc = base.drop_duplicates("accession").groupby("symbol").size()
        rc = rc.reindex(symbols).fillna(0.0)
        bc = bc.reindex(symbols).fillna(0.0) * (90.0 / 365.0)
        return np.log1p(rc) - np.log1p(bc)


register(OfficerChange())
register(NonReliance())
register(EightKFreqAnomaly())
