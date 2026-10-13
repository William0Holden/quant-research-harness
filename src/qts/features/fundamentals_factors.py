"""Fundamental factor providers: value (earnings yield), quality (gross profitability),
and net share issuance — all PIT via filing-dated fundamentals."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..pit.store import PITStore
from ..types import Backtestability, Direction, FeatureSpec
from .base import FeatureProvider, register


def _latest_close(store: PITStore, symbols: list[str], as_of: date) -> pd.Series:
    df = store.prices_as_of(symbols, as_of, lookback_days=10)
    if df.empty:
        return pd.Series(np.nan, index=symbols)
    last = df.sort_values("dt").groupby("symbol")["close"].last()
    return last.reindex(symbols)


def _concept(store, symbols, concepts, as_of, annual_only=False) -> pd.Series:
    """First available concept among `concepts` per symbol -> value Series."""
    df = store.fundamentals_as_of(symbols, concepts, as_of, annual_only=annual_only)
    if df.empty:
        return pd.Series(np.nan, index=symbols)
    # prefer concepts in given order
    order = {c: i for i, c in enumerate(concepts)}
    df["pref"] = df["concept"].map(order)
    df = df.sort_values("pref").groupby("symbol")["value"].first()
    return df.reindex(symbols)


class EarningsYield(FeatureProvider):
    spec = FeatureSpec(
        name="earnings_yield",
        direction=Direction.HIGHER_BETTER,
        backtestability=Backtestability.BACKTESTABLE,
        description="Annual NetIncomeLoss / market cap (shares_out * close), PIT.",
    )

    def compute(self, store, symbols, as_of):
        ni = _concept(store, symbols, ["NetIncomeLoss"], as_of, annual_only=True)
        sh = _concept(store, symbols, ["CommonStockSharesOutstanding"], as_of)
        px = _latest_close(store, symbols, as_of)
        mktcap = sh * px
        ey = ni / mktcap.replace(0, np.nan)
        return ey.reindex(symbols)


class GrossProfitability(FeatureProvider):
    spec = FeatureSpec(
        name="gross_profitability",
        direction=Direction.HIGHER_BETTER,
        backtestability=Backtestability.BACKTESTABLE,
        description="(Revenue - COGS) / Assets — Novy-Marx quality.",
    )

    def compute(self, store, symbols, as_of):
        gp = _concept(store, symbols, ["GrossProfit"], as_of, annual_only=True)
        rev = _concept(store, symbols,
                       ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
                       as_of, annual_only=True)
        cogs = _concept(store, symbols,
                        ["CostOfGoodsAndServicesSold", "CostOfRevenue"],
                        as_of, annual_only=True)
        assets = _concept(store, symbols, ["Assets"], as_of)
        gp = gp.fillna(rev - cogs)
        return (gp / assets.replace(0, np.nan)).reindex(symbols)


class NetShareIssuance(FeatureProvider):
    spec = FeatureSpec(
        name="net_issuance",
        direction=Direction.LOWER_BETTER,
        backtestability=Backtestability.BACKTESTABLE,
        description="YoY % change in shares outstanding; issuers underperform.",
    )

    def compute(self, store, symbols, as_of):
        now = _concept(store, symbols, ["CommonStockSharesOutstanding",
                                        "WeightedAverageNumberOfDilutedSharesOutstanding"], as_of)
        prior = _concept(store, symbols, ["CommonStockSharesOutstanding",
                                          "WeightedAverageNumberOfDilutedSharesOutstanding"],
                         as_of - timedelta(days=365))
        iss = now / prior.replace(0, np.nan) - 1.0
        return iss.reindex(symbols)


register(EarningsYield())
register(GrossProfitability())
register(NetShareIssuance())
