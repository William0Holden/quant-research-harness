"""Choose the turnover controls empirically, on the book we actually trade.

Sweeps cadence x no-trade band x partial-adjustment lambda x conviction through the
SIZED backtest (the one that runs the live construction), charges the real CostModel,
and ranks by net-of-cost Sharpe. Writes outputs/turnover_study.csv.

The headline comparison is the first two rows it prints: the legacy equal-weight
simulator vs the sized book. They are different portfolios, and only the second one is
the strategy that trades.

    uv run python -m qts.research.turnover_study [--start 2022-01-01] [--quick]
"""
from __future__ import annotations

import argparse
import itertools
from datetime import date

import pandas as pd

from ..backtest.engine import run_backtest
from ..backtest.metrics import summarize
from ..config import OUTPUTS_DIR, CostModel
from ..features.base import backtestable_providers
from ..features.ranker import rank_universe
from ..pit.store import PITStore

OUT_CSV = OUTPUTS_DIR / "turnover_study.csv"
# Rankings depend only on (providers, date) -- never on the policy being swept -- so they
# survive across runs. Delete this file whenever a provider or the universe changes.
RANK_CACHE = OUTPUTS_DIR / "rank_cache.pkl"

# Cadences worth testing. Weekly is the incumbent; the slower two trade the same signals
# less often, which is only a loss if the signals decay faster than the cadence.
CADENCES = {"weekly": "W-FRI", "biweekly": "2W-FRI", "monthly": "ME"}

# Band values must be read against TYPICAL POSITION SIZE, not as round numbers. The book
# holds ~100 names at ~0.65% of equity each, so a routine per-name adjustment is tens of
# bps. A 50bp band -- 3/4 of a whole position -- freezes the entire portfolio: positions
# can never grow into their targets and the book decays into cash at a flattering Sharpe.
# That failure mode is why FIDELITY_FLOOR below exists.
BANDS = [0.0, 0.0005, 0.001, 0.002, 0.004]   # 5bp .. 40bp
LAMBDAS = [1.0, 0.8, 0.6, 0.4]               # partial adjustment toward target
DUST_W = 0.0015

# A policy must deliver the book the sizer asked for -- in BOTH directions. Configs
# outside this band are DISQUALIFIED however good their Sharpe looks.
#   Too LOW  -> the policy is sitting in cash. Sharpe rewards that, and it is not skill.
#   Too HIGH -> the policy has let the book ratchet ABOVE its vol target (cash earns
#               nothing, so a rising book drifts up while the band suppresses the trim).
#               The first sweep's "winners" ran at 1.28x intended gross and 200+ names;
#               their extra return was extra beta. trade_policy's gross guard now bounds
#               this at the source, so a breach here means the guard is not working.
FIDELITY_FLOOR = 0.95
FIDELITY_CEILING = 1.05


def _row(label, res, extra: dict) -> dict:
    p = res.periods
    net = summarize(p["long_only_net"], res.periods_per_year)
    gross = summarize(p["long_only_gross"], res.periods_per_year)
    ann_cost = p["trade_cost"].mean() * res.periods_per_year
    held = p.get("gross_exposure", pd.Series([float("nan")])).mean()
    wanted = p.get("target_gross", pd.Series([float("nan")])).mean()
    fidelity = (held / wanted) if wanted and wanted == wanted and wanted > 0 else float("nan")
    return {
        "config": label,
        **extra,
        "net_sharpe": round(net.get("sharpe", float("nan")), 3),
        "net_cagr_pct": round(net.get("cagr", float("nan")) * 100, 2),
        "gross_sharpe": round(gross.get("sharpe", float("nan")), 3),
        "max_dd_pct": round(abs(net.get("max_drawdown", float("nan"))) * 100, 2),
        "ann_cost_pct": round(ann_cost * 100, 2),
        "turnover_per_rebal_pct": round(p["turnover"].mean() * 100, 1),
        "avg_gross_exposure": round(held, 3),
        "fidelity": round(fidelity, 3),   # held gross / gross the sizer asked for
        "avg_n_positions": round(p["n_long"].mean(), 1),
        "n_periods": len(p),
    }


def run(start: date, end: date, quick: bool = False) -> pd.DataFrame:
    cost = CostModel()
    bands = [0.0, 0.005] if quick else BANDS
    lams = [1.0, 0.5] if quick else LAMBDAS
    cadences = {"weekly": "W-FRI"} if quick else CADENCES

    rows: list[dict] = []
    with PITStore(read_only=True) as store:
        provs = backtestable_providers()

        # Reference point: the simulator as it was -- equal-weighted, fully invested,
        # no sizing. Reported so the gap to the real book is on the record, not implied.
        legacy = run_backtest(store, provs, start, end, freq="W-FRI",
                              cost_model=cost, sizing=False)
        rows.append(_row("LEGACY equal-weight (not the traded book)", legacy,
                         {"cadence": "weekly", "band_w": 0.0, "lam": 1.0,
                          "conviction": False}))

        # The ranking is identical for every (band, lam, conviction) cell -- it depends
        # only on the date. Recomputing the feature panel per cell would multiply the
        # dominant cost by ~40x for no new information, so memoise it. Persisted to disk
        # because the ranking is also identical across RUNS of the study: without this, a
        # re-run after any bug fix pays the full ~2h feature-panel cost again, which is a
        # strong incentive not to re-run -- exactly the wrong incentive to have.
        memo: dict = {}
        if RANK_CACHE.exists():
            memo = pd.read_pickle(RANK_CACHE)
            print(f"  (ranking cache: {len(memo)} dates)")

        def ranked_for(_cname: str):
            def _f(as_of: date):
                if as_of not in memo:
                    syms = list(store.universe_as_of(as_of).symbol)
                    memo[as_of] = rank_universe(store, provs, syms, as_of)
                    if len(memo) % 25 == 0:
                        RANK_CACHE.parent.mkdir(parents=True, exist_ok=True)
                        pd.to_pickle(memo, RANK_CACHE)
                return memo[as_of]

            return _f

        for (cname, freq), band, lam, conv in itertools.product(
            cadences.items(), bands, lams, (False, True)
        ):
            cfg = {"conviction": True} if conv else {}
            res = run_backtest(
                store, provs, start, end, freq=freq, cost_model=cost, sizing=True,
                cfg=cfg, policy={"band_w": band, "lam": lam, "dust_w": DUST_W},
                ranked_fn=ranked_for(cname),
            )
            label = f"{cname} band={band:.4f} lam={lam:.2f} conv={int(conv)}"
            rows.append(_row(label, res, {"cadence": cname, "band_w": band,
                                          "lam": lam, "conviction": conv}))
            print(f"  {label:44s} netSharpe {rows[-1]['net_sharpe']:6.3f} "
                  f"cost {rows[-1]['ann_cost_pct']:5.2f}%/yr", flush=True)

    RANK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(memo, RANK_CACHE)

    df = pd.DataFrame(rows)
    df["eligible"] = df["fidelity"].between(FIDELITY_FLOOR, FIDELITY_CEILING)
    df = df.sort_values(["eligible", "net_sharpe"], ascending=[False, False])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    return df


def pick(df: pd.DataFrame) -> pd.Series | None:
    """Best net Sharpe among configs that actually deploy the book (fidelity >= floor).

    Ranking on Sharpe alone would hand the crown to whichever config sat in cash: a
    heavily-banded book barely trades, barely invests, and posts a lovely Sharpe on a
    2% return. That is the same illusion that makes vol-targeted noise look like alpha.
    """
    ok = df[df["eligible"] & df["config"].str.startswith(("weekly", "biweekly", "monthly"))]
    return None if ok.empty else ok.iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="turnover / cadence parameter study")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 6, 30))
    ap.add_argument("--quick", action="store_true", help="small grid, for a smoke test")
    a = ap.parse_args()

    print(f"turnover study {a.start} -> {a.end}  (cost: "
          f"{CostModel().per_side_bps():.0f}bps/side)\n")
    df = run(a.start, a.end, quick=a.quick)
    cols = ["config", "net_sharpe", "net_cagr_pct", "max_dd_pct", "ann_cost_pct",
            "turnover_per_rebal_pct", "avg_gross_exposure", "fidelity",
            "avg_n_positions", "eligible"]
    print("\n=== all configs (eligible first, then net Sharpe) ===")
    print(df[cols].to_string(index=False))

    best = pick(df)
    if best is None:
        print(f"\nNO ELIGIBLE CONFIG: nothing held {FIDELITY_FLOOR:.0%}-"
              f"{FIDELITY_CEILING:.0%} of the gross the sizer asked for. Too low = the "
              "band is wider than a typical position and froze the book into cash. Too "
              "high = the gross guard is not bounding the ratchet. Fix it; don't re-rank.")
    else:
        print(f"\n=== PICK (best net Sharpe, fidelity in "
              f"[{FIDELITY_FLOOR:.0%}, {FIDELITY_CEILING:.0%}]) ===")
        print(f"  {best['config']}")
        print(f"  net Sharpe {best['net_sharpe']}  net CAGR {best['net_cagr_pct']}%  "
              f"cost {best['ann_cost_pct']}%/yr  fidelity {best['fidelity']}")
        print(f"\n  -> config.TradePolicy: band_w={best['band_w']}, lam={best['lam']}, "
              f"dust_w={DUST_W}; conviction={bool(best['conviction'])}; "
              f"cadence={best['cadence']}")
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
