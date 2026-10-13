"""In-memory stand-in for PITStore.prices_as_of, for use inside the backtest loop.

The sizing pipeline estimates each name's trailing vol via `store.prices_as_of([sym], ...)`
-- one DuckDB round-trip PER SYMBOL PER REBALANCE. A 4-year weekly backtest over a
~100-name book is ~25,000 queries, which turns a 30-second run into a coffee break.

The engine has already pivoted every price into a single wide panel for P&L. This adapter
serves the sizer from that panel, so the backtest can call the REAL sizing code (rather
than a fast, subtly-different copy of it) without paying for it.

PIT note: the panel holds split/dividend-ADJUSTED closes, whose adjustment factors are
known only in hindsight. Vol is estimated from LOG RETURNS, which are invariant to a
constant multiplicative factor, so a retroactive adjustment leaves the vol estimate
unchanged except across the adjustment date itself. The rows are still truncated at
``as_of``, so no future price is ever visible. This is a deliberate, bounded relaxation
of PIT purity, confined to vol estimation -- ranking still goes through the real store.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

TRADING_DAYS = 252


class PanelStore:
    """Duck-types the slice of PITStore that the sizing pipeline actually uses."""

    def __init__(self, panel: pd.DataFrame):
        """panel: index=dt (DatetimeIndex), columns=symbol, values=adj_close."""
        self.panel = panel
        # Log returns for the whole universe, computed once. `annualized_vols` slices
        # this instead of re-deriving returns per symbol per rebalance.
        self._logret = np.log(panel.where(panel > 0)).diff()

    def annualized_vols(
        self, symbols: list[str], as_of: datetime | date, lookback: int = 126,
        min_obs: int = 20,
    ) -> pd.Series:
        """Vectorised equivalent of sizing.estimate_vols, for the whole book at once.

        sizing.estimate_vols opts into this when the store advertises it (PITStore does
        not). Same estimator -- trailing stdev of log returns, annualised, NaN when there
        is too little history to trust -- but one slice instead of N DuckDB round-trips.
        """
        cols = [s for s in symbols if s in self._logret.columns]
        if not cols:
            return pd.Series({s: float("nan") for s in symbols}, dtype=float)

        cutoff = pd.Timestamp(as_of).normalize()
        window = self._logret.loc[self._logret.index <= cutoff, cols].tail(lookback)
        vols = window.std(ddof=1) * np.sqrt(TRADING_DAYS)
        vols[window.count() < min_obs] = float("nan")
        return vols.reindex(symbols)

    def prices_as_of(
        self, symbols: list[str], as_of: datetime | date,
        lookback_days: int | None = None,
    ) -> pd.DataFrame:
        """Long-form [symbol, dt, adj_close] for `symbols`, truncated at `as_of`."""
        cols = [s for s in symbols if s in self.panel.columns]
        if not cols:
            return pd.DataFrame(columns=["symbol", "dt", "adj_close"])

        cutoff = pd.Timestamp(as_of).normalize()
        window = self.panel.loc[self.panel.index <= cutoff, cols]
        if lookback_days is not None:
            window = window.loc[window.index >= cutoff - pd.Timedelta(days=lookback_days)]

        out = (
            window.stack(future_stack=True)
            .rename("adj_close")
            .reset_index()
            .dropna(subset=["adj_close"])
        )
        out.columns = ["dt", "symbol", "adj_close"]
        return out[["symbol", "dt", "adj_close"]]
