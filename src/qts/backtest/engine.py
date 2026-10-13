"""Walk-forward, point-in-time backtest engine.

Discipline baked in:
- Decisions at rebalance date t use ONLY as_of=t data (rank_universe -> prices/fundamentals
  with knowledge_time <= t). Execution is on the NEXT trading bar (no same-bar look-ahead).
- The P&L price panel (future prices) is used ONLY to score already-decided positions,
  never to rank.
- Costs (spread+slippage+commission) and short borrow are MANDATORY and non-zero by
  default (CostModel). Turnover is charged every rebalance.
- Leak guard #6: forward_only providers are refused; only backtestable providers run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from ..config import CostModel, TradePolicy
from ..risk.portfolio import build_target_portfolio
from ..risk.regime import prefetch_series
from ..risk.trade_policy import apply_trade_policy
from .panel_store import PanelStore

# Weights are scale-free, so the sized path only needs a nominal equity to size against.
NOMINAL_EQUITY = 100_000.0

from ..features.base import FeatureProvider
from ..features.ranker import rank_universe
from ..pit.store import PITStore
from ..types import Backtestability


@dataclass
class BacktestResult:
    periods: pd.DataFrame            # index=rebalance date; long_only/long_short/benchmark net rets
    positions: dict                 # date -> {long:[...], short:[...]}
    schedule: list                  # rebalance dates used
    periods_per_year: float
    cost_model: CostModel
    weights_history: dict = field(default_factory=dict)


def _trading_days(store: PITStore) -> np.ndarray:
    df = store.con.execute("SELECT DISTINCT dt FROM prices ORDER BY dt").df()
    return pd.to_datetime(df["dt"]).values


def _adj_panel(store: PITStore) -> pd.DataFrame:
    df = store.con.execute("SELECT symbol, dt, adj_close FROM prices").df()
    piv = df.pivot_table(index="dt", columns="symbol", values="adj_close").sort_index()
    piv.index = pd.to_datetime(piv.index)
    return piv


def make_schedule(days: np.ndarray, start: date, end: date, freq: str = "W-FRI") -> list:
    days = pd.to_datetime(days)
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    if len(days) == 0:
        return []
    cal = pd.date_range(days.min(), days.max(), freq=freq)
    # snap each scheduled date to the last actual trading day <= it
    sched = []
    for d in cal:
        prior = days[days <= d]
        if len(prior):
            sched.append(prior[-1])
    return sorted(set(sched))


def _next_bar(panel_index: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp | None:
    after = panel_index[panel_index > t]
    return after[0] if len(after) else None


def run_backtest(
    store: PITStore,
    providers: list[FeatureProvider],
    start: date,
    end: date,
    freq: str = "W-FRI",
    n_buckets: int = 5,
    cost_model: CostModel | None = None,
    weights: dict[str, float] | None = None,
    weights_fn: Callable[[date], dict | None] | None = None,
    sizing: bool = True,
    cfg: dict | None = None,
    policy: dict | None = None,
    regime_on: bool = True,
    ranked_fn: Callable[[date], pd.DataFrame | None] | None = None,
) -> BacktestResult:
    """Walk-forward backtest of the LIVE portfolio construction.

    ``sizing=True`` (the default) runs the same `build_target_portfolio` + `trade_policy`
    that `exec.rebalance` runs: inverse-vol / conviction weights, vol targeting, the
    regime brake, position caps, and the turnover controls. long_only_* is then the
    return of that actual book (w . r, with the un-deployed remainder in cash).

    ``sizing=False`` reproduces the legacy behaviour: an equal-weighted, fully-invested
    top-bucket basket. That is NOT the strategy that trades -- it ignores sizing entirely
    and so reports the returns of a portfolio nobody owns. Kept only so the old
    equal-weight reference can still be produced on demand.

    ``policy``: kwargs for `apply_trade_policy` (band_w / lam / dust_w). Defaults to
    config.TradePolicy. ``cfg``: sizing cfg (target_vol, conviction, caps...).

    ``ranked_fn``: supply the ranked frame for a date instead of computing it. The
    ranking depends only on (providers, weights, date) -- NOT on the sizing cfg or the
    trade policy -- so a parameter sweep re-deriving it per config would recompute the
    identical, expensive feature panel for every cell of the grid. Callers that hold a
    ranking constant across configs pass a memo here. Must return the same frame
    rank_universe would; None means "skip this date".
    """
    cost_model = cost_model or CostModel()
    policy = TradePolicy().as_kwargs() if policy is None else policy

    # Leak guard #6: refuse forward-only providers in backtest mode.
    bad = [p.spec.name for p in providers
           if p.spec.backtestability != Backtestability.BACKTESTABLE]
    if bad:
        raise ValueError(f"forward_only providers refused in backtest mode: {bad}")

    days = _trading_days(store)
    panel = _adj_panel(store)
    schedule = make_schedule(days, start, end, freq)
    if len(schedule) < 3:
        raise ValueError("schedule too short; widen the date range")

    per_side = cost_model.per_side_bps() / 1e4
    rows = []
    positions: dict = {}
    weights_hist: dict = {}
    prev_w = pd.Series(dtype=float)   # long-only weights carried for turnover

    # The sized path calls the real portfolio builder once per rebalance. Two things
    # would make that unaffordable if done naively, so both are hoisted out of the loop:
    #   - vols: served from the in-memory panel, not one DuckDB query per symbol per bar
    #   - regime: FRED fetched once for the whole run, not once per rebalance date
    sizer_store = PanelStore(panel) if sizing else None
    regime_series = prefetch_series(end) if (sizing and regime_on) else None

    for i in range(len(schedule) - 1):
        t = pd.Timestamp(schedule[i])
        t_next = pd.Timestamp(schedule[i + 1])
        entry = _next_bar(panel.index, t)        # execute AFTER decision date
        exit_ = _next_bar(panel.index, t_next)
        if entry is None or exit_ is None:
            continue

        syms = list(store.universe_as_of(t.date()).symbol)   # also the benchmark basket
        if ranked_fn is not None:
            ranked = ranked_fn(t.date())
            if ranked is None:
                continue
        else:
            w_t = weights
            if weights_fn is not None:
                wf = weights_fn(t.date())
                w_t = weights if wf is None else wf
            ranked = rank_universe(store, providers, syms, t.date(), n_buckets, w_t)
        ranked = ranked.dropna(subset=["bucket"])
        if ranked.empty:
            continue

        longs = list(ranked.index[ranked["bucket"] == n_buckets])
        shorts = list(ranked.index[ranked["bucket"] == 1])
        positions[t.date().isoformat()] = {"long": longs, "short": shorts}

        # forward return per name over the holding period (decided names only)
        def _ret(names):
            names = [n for n in names if n in panel.columns]
            if not names:
                return 0.0, pd.Series(dtype=float)
            p0 = panel.loc[entry, names]
            p1 = panel.loc[exit_, names]
            r = (p1 / p0 - 1.0).dropna()
            return (r.mean() if len(r) else 0.0), r

        long_ret, long_legs = _ret(longs)
        short_ret, _ = _ret(shorts)
        bench_syms = [n for n in syms if n in panel.columns]
        bench_ret = (panel.loc[exit_, bench_syms] / panel.loc[entry, bench_syms] - 1.0).dropna().mean()

        # ---- portfolio construction + costs ----
        if sizing:
            # The SAME builder exec/rebalance.py calls, then the SAME trade policy.
            target_w, sz_diag = build_target_portfolio(
                ranked, sizer_store, t.date(), NOMINAL_EQUITY,
                regime_on=regime_on, cfg=cfg, regime_series=regime_series,
                prev_w=prev_w.to_dict(),
            )
            traded_w, pol_diag = apply_trade_policy(prev_w.to_dict(), target_w, **policy)
            w_new = pd.Series(traded_w, dtype=float)
            turnover = float(pol_diag["turnover"])

            # Held names are NOT always this period's top bucket: the band and partial
            # adjustment deliberately carry names the ranker has since dropped. Score
            # the book we actually hold, not the bucket we just picked.
            _, held_r = _ret(list(w_new.index))
            held_r = held_r.reindex(w_new.index).fillna(0.0)
            port_ret = float((w_new * held_r).sum())  # un-deployed remainder sits in cash
            gross_exposure = float(w_new.sum())
        else:
            # Legacy reference book: equal-weighted, fully invested, no sizing at all.
            w_new = (pd.Series(1.0 / len(longs), index=longs) if longs
                     else pd.Series(dtype=float))
            turnover = float(w_new.subtract(prev_w, fill_value=0.0).abs().sum())
            held_r = long_legs.reindex(w_new.index).fillna(0.0)
            port_ret = long_ret
            gross_exposure = float(w_new.sum())
            pol_diag = {}

        trade_cost = turnover * per_side
        weights_hist[t.date().isoformat()] = w_new.to_dict()

        # Carry positions to the next rebalance with PRICE DRIFT applied: a name that
        # doubled is a bigger slice of the book next period even if we never traded it.
        # Skipping this would understate turnover and flatter the no-trade band, which
        # exists precisely to absorb drift.
        denom = 1.0 + port_ret
        prev_w = (w_new * (1.0 + held_r) / denom) if denom > 0 else w_new

        hold_days = (exit_ - entry).days
        borrow = cost_model.borrow_bps_annual / 1e4 * (hold_days / 365.0)  # on short notional

        long_only_net = port_ret - trade_cost
        # long/short: equal-weight legs. NOT the traded book (production is long-only) --
        # a research reference for the short leg's raw information, kept unsized.
        long_short_net = long_ret - short_ret - 2 * trade_cost - borrow

        rows.append({
            "date": t.date(), "entry": entry.date(), "exit": exit_.date(),
            "long_only_gross": port_ret, "long_only_net": long_only_net,
            "long_short_gross": long_ret - short_ret, "long_short_net": long_short_net,
            "benchmark": bench_ret, "turnover": turnover,
            "trade_cost": trade_cost, "borrow_cost": borrow,
            "n_long": len(w_new), "n_short": len(shorts),
            "gross_exposure": gross_exposure,
            # What the sizer WANTED to hold, before the turnover controls. If a policy
            # systematically fails to reach this, it is not saving costs -- it is quietly
            # de-risking the book, and its Sharpe is flattered by sitting in cash.
            "target_gross": pol_diag.get("target_gross", gross_exposure),
            "n_banded": pol_diag.get("n_banded", 0),
            "n_dust_exits": pol_diag.get("n_dust_exits", 0),
        })

    periods = pd.DataFrame(rows).set_index("date")
    ppy = _periods_per_year(freq)
    return BacktestResult(periods, positions, schedule, ppy, cost_model, weights_hist)


def _periods_per_year(freq: str) -> float:
    """Rebalances per year for a pandas offset alias, INCLUDING its multiplier.

    The multiplier is not cosmetic. '2W-FRI' used to fall through every branch (it starts
    with '2', not 'W') and silently return 52.0 -- so a biweekly backtest was annualised
    as if it rebalanced weekly, inflating its Sharpe by sqrt(2) and roughly doubling its
    CAGR. A cadence study built on that would 'prove' that trading less often earns more,
    which is an artefact of the divisor, not a fact about the market.
    """
    f = freq.upper().strip()
    m = re.match(r"^(\d*)\s*([A-Z]+)", f)
    n = int(m.group(1)) if (m and m.group(1)) else 1
    base = (m.group(2) if m else f)

    if base.startswith(("BM", "BQ", "BY", "BA")):
        base = base[1:]          # business-month/quarter/year: same count as calendar
    if base.startswith(("Y", "A")):
        per = 1.0
    elif base.startswith("Q"):
        per = 4.0
    elif base.startswith("M"):   # M, ME, MS
        per = 12.0
    elif base.startswith("W"):
        per = 52.0
    elif base.startswith(("B", "D", "C")):
        per = 252.0
    else:
        per = 52.0
    return per / n
