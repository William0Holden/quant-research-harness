"""Performance metrics from a periodic net-return series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(period_returns: pd.Series) -> pd.Series:
    return (1.0 + period_returns.fillna(0.0)).cumprod()


def summarize(period_returns: pd.Series, periods_per_year: float) -> dict:
    r = period_returns.dropna()
    if len(r) < 2:
        return {k: np.nan for k in
                ("cagr", "vol", "sharpe", "max_drawdown", "hit_rate", "n_periods")}
    eq = equity_curve(r)
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = r.std() * np.sqrt(periods_per_year)
    sharpe = (r.mean() * periods_per_year) / vol if vol else np.nan
    dd = (eq / eq.cummax() - 1.0).min()
    return {
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
        "hit_rate": float((r > 0).mean()),
        "n_periods": int(len(r)),
    }
