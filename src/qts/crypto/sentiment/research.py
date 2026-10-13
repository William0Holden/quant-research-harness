"""Risk-IC measurement: does each signal predict RISK (not return)?

For each signal, walk the accrued history, pair each day's signal value with the realized
volatility / drawdown over the FOLLOWING window, and compute a rank information coefficient
with a t-stat and sample size. A signal earns un-gating only on this record -- the same
discipline the equity providers earn theirs on, but graded against risk, not return.

Grading targets:
  - Signal 1 (vol_forecast)   -> forward realized vol.  Also reported vs the trailing-vol
                                 baseline, so "beats doing nothing" is explicit.
  - Signal 2 (equity risk app) -> forward crypto drawdown (a DROP should precede drawdowns).
  - Signal 3 (attention spike) -> forward realized vol (a spike should precede higher vol).

Returns "insufficient history" until enough non-NaN pairs exist -- never a spurious IC on
three data points. Today that is the expected answer everywhere: the crypto sentiment tables
are empty and must accrue for weeks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_PAIRS = 20   # below this, report insufficient rather than a number


@dataclass
class ICResult:
    signal: str
    target: str
    n: int
    ic: float = float("nan")
    t_stat: float = float("nan")
    note: str = ""

    def as_dict(self) -> dict:
        return {"signal": self.signal, "target": self.target, "n": self.n,
                "ic": None if not np.isfinite(self.ic) else round(self.ic, 4),
                "t_stat": None if not np.isfinite(self.t_stat) else round(self.t_stat, 2),
                "note": self.note}


def rank_ic(signal: pd.Series, target: pd.Series, name: str, target_name: str) -> ICResult:
    """Spearman IC between a signal and a forward-risk target, aligned on date.

    Higher-signal -> higher-risk is a POSITIVE IC. For signal 2 the hypothesis is that a
    risk-appetite DROP precedes a WORSE (more negative) drawdown, i.e. lower signal -> lower
    (more negative) target -> also a positive IC on the raw pair. Sign interpretation is left
    to the caller; this returns the raw rank correlation and its t-stat.
    """
    df = pd.concat([pd.Series(signal, name="s"), pd.Series(target, name="t")], axis=1).dropna()
    n = len(df)
    if n < MIN_PAIRS:
        return ICResult(name, target_name, n, note=f"insufficient history (n={n}<{MIN_PAIRS})")
    ic = df["s"].rank().corr(df["t"].rank())
    if not np.isfinite(ic):
        return ICResult(name, target_name, n, note="degenerate (no variance)")
    t = ic * np.sqrt(max(n - 2, 1) / max(1 - ic * ic, 1e-9))
    return ICResult(name, target_name, n, ic=float(ic), t_stat=float(t))


def evaluate(signal_by_date: dict, target_by_date: dict, name: str,
             target_name: str) -> ICResult:
    """Convenience: build aligned Series from {date: value} dicts and score them."""
    s = pd.Series(signal_by_date, dtype=float)
    t = pd.Series(target_by_date, dtype=float)
    s.index = pd.to_datetime(s.index)
    t.index = pd.to_datetime(t.index)
    return rank_ic(s, t, name, target_name)


@dataclass
class SignalScorecard:
    """The full accrued risk-IC picture across all three signals, for the dashboard + gate."""
    results: list[ICResult] = field(default_factory=list)

    def add(self, r: ICResult) -> None:
        self.results.append(r)

    def earned(self, min_t: float = 2.0) -> list[str]:
        """Signals whose risk-IC clears |t| >= min_t on any target. These, and only these,
        are candidates to un-gate -- and even then only by a deliberate, separate decision."""
        return sorted({r.signal for r in self.results
                       if np.isfinite(r.t_stat) and abs(r.t_stat) >= min_t})

    def as_records(self) -> list[dict]:
        return [r.as_dict() for r in self.results]
