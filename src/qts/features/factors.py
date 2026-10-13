"""Price-based factor providers: momentum (12-1) and low-volatility."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..pit.store import PITStore
from ..types import Backtestability, Direction, FeatureSpec
from .base import FeatureProvider, register


def _price_panel(store: PITStore, symbols: list[str], as_of: date, lookback_days: int) -> pd.DataFrame:
    df = store.prices_as_of(symbols, as_of, lookback_days=lookback_days)
    if df.empty:
        return pd.DataFrame()
    piv = df.pivot_table(index="dt", columns="symbol", values="adj_close")
    return piv.sort_index()


class Momentum12_1(FeatureProvider):
    spec = FeatureSpec(
        name="momentum_12_1",
        direction=Direction.HIGHER_BETTER,
        backtestability=Backtestability.BACKTESTABLE,
        description="Total return from t-252 to t-21 (skip most-recent month).",
    )

    def compute(self, store, symbols, as_of):
        piv = _price_panel(store, symbols, as_of, lookback_days=420)
        if piv.empty or len(piv) < 60:
            return pd.Series(np.nan, index=symbols)
        # most recent row = t; skip last ~21 trading days
        end = piv.iloc[-21] if len(piv) > 21 else piv.iloc[-1]
        start = piv.iloc[-252] if len(piv) >= 252 else piv.iloc[0]
        mom = end / start - 1.0
        return mom.reindex(symbols)


class LowVolatility(FeatureProvider):
    spec = FeatureSpec(
        name="low_volatility",
        direction=Direction.LOWER_BETTER,
        backtestability=Backtestability.BACKTESTABLE,
        description="Annualized stdev of daily returns over trailing ~6 months.",
    )

    def compute(self, store, symbols, as_of):
        piv = _price_panel(store, symbols, as_of, lookback_days=200)
        if piv.empty or len(piv) < 40:
            return pd.Series(np.nan, index=symbols)
        rets = piv.pct_change().iloc[-126:]
        vol = rets.std() * np.sqrt(252)
        return vol.reindex(symbols)


register(Momentum12_1())
register(LowVolatility())
