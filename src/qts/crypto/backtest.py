"""Backtest the trend book, net of REAL costs, against the only benchmark that matters.

BENCHMARK = BTC BUY & HOLD (Sharpe 0.54, maxDD -76.7%). Not cash. Not equal-weight crypto.
A long-only crypto book in a bull market will make money and it will mean nothing; the only
honest question is whether it beats simply holding Bitcoin.

The claim this backtest is here to test is NOT "we have alpha". It is: "trend timing gets
roughly buy-and-hold returns at roughly half the drawdown." That is a DRAWDOWN claim, so
maxDD and Calmar are the headline, and Sharpe is reported with its (large) error bar.

SAMPLE-SIZE HONESTY: 5.5 years of daily bars is ~1 crypto cycle and ~4 effective independent
regime observations. The single-trial significance threshold at this length is Sharpe >= 0.84
-- which BTC's own 0.54 does not clear. Nothing here can establish skill. It can only falsify.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..risk.trade_policy import apply_trade_policy
from .costs import CryptoCostModel
from .prices import fetch_panel
from .sizing import VOL_TARGET, build_crypto_targets
from .trend import WARMUP, exposure
from .universe import HALF_SPREAD_BPS, UNIVERSE

ANN = 365


def stats(r: pd.Series) -> dict:
    r = pd.Series(r).dropna()
    if r.empty:
        return {}
    nav = (1 + r).cumprod()
    yrs = len(r) / ANN
    cagr = (nav.iloc[-1] ** (1 / yrs) - 1) if nav.iloc[-1] > 0 else -1.0
    sharpe = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else 0.0
    dd = float((nav / nav.cummax() - 1).min())
    # SE(Sharpe) ~ sqrt((1 + SR^2/2)/T) -- the honest error bar. At 5.5y this is ~0.44.
    se = float(np.sqrt((1 + sharpe**2 / 2) / len(r)) * np.sqrt(ANN))
    return {"cagr": float(cagr), "sharpe": float(sharpe), "sharpe_se": se,
            "max_dd": dd, "calmar": float(cagr / abs(dd)) if dd else 0.0, "n": len(r)}


def run(panel: pd.DataFrame, cost: CryptoCostModel, liquidity: str = "taker",
        vol_target: float = VOL_TARGET, overlay_on: bool = False,
        band_w: float = 0.0, lam: float = 1.0) -> dict:
    """Daily-evaluated trend book. Decide on bar t, hold t+1 (no same-bar look-ahead).

    band_w is the SAME no-trade band the equity book uses (risk/trade_policy.py), and it
    matters far more here: Garleanu-Pedersen says the optimal band widens with cost, and
    crypto costs 30-64bps/side against equities' 8. Without it the ensemble makes tiny
    daily gross adjustments and pays a spread on every single one of them.
    """
    R = panel.pct_change()
    exps = {s: exposure(panel[s]) for s in panel.columns}
    cost_bps = {s: (cost.maker_bps(HALF_SPREAD_BPS.get(s, 20.0)) if liquidity == "maker"
                    else cost.taker_bps(HALF_SPREAD_BPS.get(s, 20.0)))
                for s in panel.columns}

    prev: dict[str, float] = {}
    rows = []
    for i in range(WARMUP, len(panel) - 1):
        d, nxt = panel.index[i], panel.index[i + 1]
        te = {s: float(exps[s].loc[d]) if d in exps[s].index else float("nan")
              for s in panel.columns}
        closes = {s: panel[s].iloc[: i + 1] for s in panel.columns}   # PIT: only <= d
        w, diag = build_crypto_targets(te, closes, sent_z=None, vol_target=vol_target,
                                       overlay_on=overlay_on)
        # Suppress trades not worth making. max_gross_drift=None: gross is already hard-
        # capped at 1.0 upstream by the sizer, so the equity book's cash-funded-ratchet
        # guard has nothing left to do here.
        w, _pol = apply_trade_policy(prev, w, band_w=band_w, lam=lam, dust_w=0.0,
                                     max_gross_drift=None)

        turn = sum(abs(w.get(s, 0.0) - prev.get(s, 0.0)) for s in panel.columns)
        c = sum(abs(w.get(s, 0.0) - prev.get(s, 0.0)) * cost_bps[s] / 1e4
                for s in panel.columns)
        fr = R.loc[nxt]
        port = sum(w.get(s, 0.0) * float(fr.get(s, 0.0) or 0.0) for s in panel.columns)
        rows.append({"dt": nxt, "gross": diag["gross"], "turnover": turn,
                     "cost": c, "gross_ret": port, "net_ret": port - c})
        # carry with price drift: a held position's weight moves with its own return
        prev = {s: w.get(s, 0.0) * (1 + float(fr.get(s, 0.0) or 0.0)) / (1 + port)
                for s in panel.columns if w.get(s, 0.0) > 0} if (1 + port) > 0 else {}

    df = pd.DataFrame(rows).set_index("dt")
    out = {"net": stats(df["net_ret"]), "gross": stats(df["gross_ret"]),
           "ann_cost_pct": float(df["cost"].mean() * ANN * 100),
           "avg_gross": float(df["gross"].mean()),
           "turnover_per_day_pct": float(df["turnover"].mean() * 100),
           "switches_per_yr": float((df["turnover"] > 0.05).sum() / (len(df) / ANN))}
    out["periods"] = df
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="crypto trend book backtest")
    ap.add_argument("--liquidity", choices=["maker", "taker"], default="taker")
    ap.add_argument("--vol-target", type=float, default=VOL_TARGET)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--band", type=float, default=0.02,
                    help="no-trade band in weight units; crypto costs 4-7x equities so the "
                         "optimal band is far wider than the equity book's 5bp")
    a = ap.parse_args()

    panel = fetch_panel(UNIVERSE)
    cost = CryptoCostModel()
    print(f"panel {panel.shape[0]} days x {panel.shape[1]} coins "
          f"({panel.index[0].date()} -> {panel.index[-1].date()})")
    print(f"cost: {a.liquidity}  vol_target {a.vol_target:.0%}  overlay {a.overlay}\n")

    res = run(panel, cost, liquidity=a.liquidity, vol_target=a.vol_target,
              overlay_on=a.overlay, band_w=a.band)
    n = res["net"]

    R = panel.pct_change()
    bh_btc = stats(R["BTC/USD"].iloc[WARMUP:])
    bh_ew = stats(R.mean(axis=1).iloc[WARMUP:])

    hdr = f"{'strategy':<28s} {'CAGR':>8s} {'Sharpe':>8s} {'+/-':>6s} {'maxDD':>8s} {'Calmar':>7s}"
    print(hdr); print("-" * len(hdr))
    for lbl, s in (("TREND BOOK (net)", n),
                   ("BTC buy & hold  <- THE BAR", bh_btc),
                   ("equal-weight buy & hold", bh_ew)):
        print(f"{lbl:<28s} {s['cagr']*100:7.1f}% {s['sharpe']:8.2f} "
              f"{s['sharpe_se']:6.2f} {s['max_dd']*100:7.1f}% {s['calmar']:7.2f}")

    print(f"\ncost {res['ann_cost_pct']:.2f}%/yr   avg gross {res['avg_gross']:.3f}   "
          f"turnover {res['turnover_per_day_pct']:.2f}%/day   "
          f"~{res['switches_per_yr']:.0f} material switches/yr")

    dd_better = n["max_dd"] > bh_btc["max_dd"]
    print(f"\nPRIMARY CLAIM (drawdown below buy & hold): "
          f"{'PASS' if dd_better else 'FAIL'}  "
          f"({n['max_dd']*100:.1f}% vs {bh_btc['max_dd']*100:.1f}%)")
    gap = n["sharpe"] - bh_btc["sharpe"]
    print(f"SECONDARY (Sharpe >= buy & hold): "
          f"{'PASS' if gap >= 0 else 'FAIL'}  (gap {gap:+.2f}, SE ~{n['sharpe_se']:.2f} "
          f"-> {'NOT significant' if abs(gap) < 2*n['sharpe_se'] else 'significant'})")


if __name__ == "__main__":
    main()
