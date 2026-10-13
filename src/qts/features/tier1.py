"""Tier-1 event/positioning providers: insider activity, 8-K intensity + restatement
flag, and short interest (optional)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..pit.store import PITStore
from ..types import Backtestability, Direction, FeatureSpec
from .base import FeatureProvider, register


def _event_counts(store, symbols, form, as_of, window_days, item=None) -> pd.Series:
    start = as_of - timedelta(days=window_days)
    df = store.events_between(symbols, form, start, as_of)
    if df.empty:
        return pd.Series(0.0, index=symbols)
    if item is not None:
        df = df[df["item"] == item]
    c = df.groupby("symbol").size().astype(float)
    return c.reindex(symbols).fillna(0.0)


class InsiderActivity(FeatureProvider):
    spec = FeatureSpec(
        name="insider_activity",
        direction=Direction.HIGHER_BETTER,  # IC-weighting decides sign empirically
        backtestability=Backtestability.BACKTESTABLE,
        description="Trailing-90d Form-4 filing intensity (coarse proxy; P/S direction "
                    "parse is a documented follow-on). IC-weighting down-weights if noisy.",
    )

    def compute(self, store, symbols, as_of):
        return _event_counts(store, symbols, "4", as_of, 90)


class EightKIntensity(FeatureProvider):
    spec = FeatureSpec(
        name="event8k_intensity",
        direction=Direction.HIGHER_BETTER,
        backtestability=Backtestability.BACKTESTABLE,
        description="Trailing-90d 8-K material-event count.",
    )

    def compute(self, store, symbols, as_of):
        return _event_counts(store, symbols, "8-K", as_of, 90)


class RestatementFlag(FeatureProvider):
    spec = FeatureSpec(
        name="restatement_flag",
        direction=Direction.LOWER_BETTER,
        backtestability=Backtestability.BACKTESTABLE,
        description="Trailing-1y count of 8-K item 4.02 (non-reliance / restatement) — red flag.",
    )

    def compute(self, store, symbols, as_of):
        return _event_counts(store, symbols, "8-K", as_of, 365, item="4.02")


class ShortInterest(FeatureProvider):
    spec = FeatureSpec(
        name="short_interest",
        direction=Direction.LOWER_BETTER,
        backtestability=Backtestability.BACKTESTABLE,
        description="Short-interest ratio (days-to-cover). Optional: returns NaN until the "
                    "short_interest table is populated from FINRA, so it self-excludes.",
    )

    def compute(self, store, symbols, as_of):
        # Optional source; absent in v1 default -> all-NaN -> excluded by the ranker.
        return pd.Series(np.nan, index=symbols)


register(InsiderActivity())
register(EightKIntensity())
register(RestatementFlag())
register(ShortInterest())
