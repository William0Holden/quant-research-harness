"""Where the equity book's turnover comes from, and what stops it -- measured.

The Jul-Aug 2026 live book swapped 11-41 of its 101 names EVERY WEEK and lost money to
churn, not to stock picking (-$673 realised on 114 round-tripped names). Two candidate
causes, each with a lever:

  1. Adaptive provider weights swing week to week (momentum 0.447 -> 0.043 -> 0.149),
     so the ranking itself reshuffles.      Lever: smoother IC blend (window, shrink).
  2. The top-quintile boundary is crossed on rank noise alone.
                                              Lever: membership hysteresis (hold_bucket).

Runs the SIZED backtest (the live construction) on the cached feature panel, re-scoring
the cached z-columns for each weight scheme so no feature is recomputed. Reports
net-of-cost Sharpe, cost, turnover, fidelity, AND the weekly name overlap -- the number
that actually describes churn. Writes outputs/churn_study.csv.

    uv run python -m qts.research.churn_study [--start 2022-01-01]
"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from ..backtest.engine import run_backtest
from ..backtest.metrics import summarize
from ..config import OUTPUTS_DIR, CostModel, TradePolicy, live_gated
from ..features.base import backtestable_providers
from ..features.ranker import assign_buckets, composite_score
from ..pit.store import PITStore
from ..research.ic import blended_ic_weights
from ..research.ic_snapshot import load_ic_history
from .turnover_study import RANK_CACHE, _row

OUT_CSV = OUTPUTS_DIR / "churn_study.csv"

# (label, ic-weight scheme or None for equal, hold_bucket)
CONFIGS = [
    ("equal weights                 | no hysteresis", None, None),
    ("adaptive 13/52 shrink .50 LIVE| no hysteresis", (13, 52, 0.50), None),
    ("adaptive 26/52 shrink .75     | no hysteresis", (26, 52, 0.75), None),
    ("adaptive 52/52 shrink .75     | no hysteresis", (52, 52, 0.75), None),
    ("equal weights                 | hold_bucket=4", None, 4),
    ("equal weights                 | hold_bucket=3", None, 3),
    ("adaptive 13/52 shrink .50 LIVE| hold_bucket=4", (13, 52, 0.50), 4),
    ("adaptive 26/52 shrink .75     | hold_bucket=4", (26, 52, 0.75), 4),
]


def _rescore(frame: pd.DataFrame, live_names: list[str], weights: dict | None) -> pd.DataFrame:
    """Recompute score/bucket from cached z-columns restricted to the LIVE provider set."""
    zcols = {f"z_{n}": n for n in live_names if f"z_{n}" in frame.columns}
    zdf = frame[list(zcols)].rename(columns=zcols)
    score = composite_score(zdf, weights)
    score = score.where(frame["score"].notna())      # keep the cached eligibility mask
    out = frame.copy()
    out["score"] = score
    out["bucket"] = assign_buckets(score, 5)
    return out.sort_values("score", ascending=False)


def _overlap(weights_hist: dict) -> float:
    keys = sorted(weights_hist)
    ov = []
    for a, b in zip(keys, keys[1:]):
        sa = {k for k, v in weights_hist[a].items() if v > 0}
        sb = {k for k, v in weights_hist[b].items() if v > 0}
        if sb:
            ov.append(len(sa & sb) / len(sb))
    return float(sum(ov) / len(ov)) if ov else float("nan")


def run(start: date, end: date | None = None) -> pd.DataFrame:
    memo: dict = pd.read_pickle(RANK_CACHE)
    dates = sorted(memo)
    end = end or dates[-1]
    gated = live_gated()
    provs = [p for p in backtestable_providers() if p.name not in gated]
    live_names = [p.name for p in provs]
    ic_hist = load_ic_history()
    cost = CostModel()
    policy = TradePolicy().as_kwargs()
    print(f"ranking cache {dates[0]}..{dates[-1]} ({len(dates)} dates); live providers: {live_names}")
    print(f"ic history {ic_hist.index.min().date()}..{ic_hist.index.max().date()} ({len(ic_hist)} rows)")

    rows = []
    with PITStore(read_only=True) as store:
        for label, scheme, hold in CONFIGS:
            rescored: dict = {}

            def ranked_for(as_of: date, scheme=scheme, rescored=rescored):
                if as_of not in memo:
                    return None
                if as_of not in rescored:
                    w = None
                    if scheme is not None and not ic_hist.empty:
                        sw, lw, sh = scheme
                        w, _ = blended_ic_weights(ic_hist, as_of, live_names,
                                                  short_window=sw, long_window=lw, shrink=sh)
                    rescored[as_of] = _rescore(memo[as_of], live_names, w)
                return rescored[as_of]

            cfg = {"hold_bucket": hold} if hold else {}
            res = run_backtest(store, provs, start, end, freq="W-FRI", cost_model=cost,
                               sizing=True, cfg=cfg, policy=policy, ranked_fn=ranked_for)
            row = _row(label, res, {"scheme": scheme, "hold_bucket": hold})
            row["name_overlap"] = round(_overlap(res.weights_history), 3)
            # The adaptive schemes only differ from equal once IC history exists
            # (2025-01+). Report that sub-window so like is compared with like.
            sub = res.periods[res.periods.index >= date(2025, 2, 1)]
            s = summarize(sub["long_only_net"], res.periods_per_year)
            row["sub2025_net_sharpe"] = round(s["sharpe"], 3)
            row["sub2025_net_cagr_pct"] = round(s["cagr"] * 100, 2)
            row["sub2025_turnover_pct"] = round(sub["turnover"].mean() * 100, 1)
            rows.append(row)
            print(f"  {label:46s} netSharpe {row['net_sharpe']:6.3f}  CAGR {row['net_cagr_pct']:5.2f}%  "
                  f"cost {row['ann_cost_pct']:4.2f}%/yr  turn/rebal {row['turnover_per_rebal_pct']:4.1f}%  "
                  f"overlap {row['name_overlap']:.3f}  fid {row['fidelity']:.3f}  n {row['avg_n_positions']:.0f} | "
                  f"2025+: Sharpe {row['sub2025_net_sharpe']:6.3f} turn {row['sub2025_turnover_pct']:4.1f}%",
                  flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 1))
    a = ap.parse_args()
    df = run(a.start)
    print(df.to_string())


if __name__ == "__main__":
    main()
