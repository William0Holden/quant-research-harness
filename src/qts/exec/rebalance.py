"""Weekly paper-rebalance runner: rank -> size -> broker.rebalance, audited to JSON.

Every run writes outputs/rebalances/<as_of>.json (weights, orders, diagnostics,
equity). The KillSwitch reads the recorded equity history: if peak-to-trough
drawdown ever breached the threshold, targets flatten to all-cash and stay flat
(reset = deliberately archive outputs/rebalances/, per the latching contract).
Records whose equity is a fabricated fallback (no broker account reachable) are
excluded from that history, so a substituted $100k never pollutes the drawdown
calculation.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

from ..config import (DEFAULT_HOLD_BUCKET, IC_LONG_WINDOW, IC_SHORT_WINDOW, IC_SHRINK,
                      OUTPUTS_DIR, TradePolicy, live_gated)
from ..features.base import backtestable_providers, default_providers
from ..features.ranker import rank_universe
from ..pit.store import PITStore
from ..research.ic import blended_ic_weights
from ..research.ic_snapshot import load_ic_history
from ..risk.limits import KillSwitch
from ..risk.portfolio import build_target_portfolio
from ..risk.trade_policy import apply_trade_policy
from .broker import AlpacaBroker, Broker, DryRunBroker, TargetPortfolio

DEFAULT_OUT_DIR = OUTPUTS_DIR / "rebalances"
MAX_DRAWDOWN = 0.25


def pick_broker() -> Broker:
    """AlpacaBroker (paper) iff both keys are in the env, else DryRunBroker."""
    if os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY"):
        return AlpacaBroker(paper=True)
    return DryRunBroker()


def _equity_history(out_dir: Path) -> list[float]:
    hist = []
    if out_dir.exists():
        for p in sorted(out_dir.glob("*.json")):
            try:
                rec = json.loads(p.read_text())
                if rec.get("equity_source") == "fallback":
                    continue
                hist.append(float(rec["equity"]))
            except Exception:  # noqa: BLE001  malformed record never blocks a run
                continue
    return hist


def run_rebalance(
    store: PITStore,
    as_of: date,
    broker: Broker,
    submit: bool = False,
    equity_fallback: float = 100_000.0,
    regime_on: bool = True,
    cfg: dict | None = None,
    out_dir: Path | None = None,
    band_w: float = 0.0,
    cadence: str = "weekly",
    policy: dict | None = None,
) -> dict:
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR

    try:
        if isinstance(broker, AlpacaBroker):
            equity = broker._account_equity()
            equity_source = "broker"
        else:
            equity = equity_fallback
            equity_source = "fallback"
    except RuntimeError:  # keys/package missing -> research equity
        equity = equity_fallback
        equity_source = "fallback"

    hist = _equity_history(out_dir)
    # A one-off fallback-equity run must never dilute real broker history: only fold the
    # current reading into the drawdown series when it's an actual broker equity, or when
    # there's no history yet to check against (a bare fallback reading is all we have).
    equity_series = hist + [equity] if (equity_source == "broker" or not hist) else hist
    kill = KillSwitch(max_drawdown=MAX_DRAWDOWN).check(equity_series)
    gated = live_gated()
    live_providers = [p for p in backtestable_providers() if p.name not in gated]
    weights_used: dict | None = None
    weights_diag: dict = {}
    untradable: dict = {}
    held_w = broker.get_positions()      # once: feeds hysteresis AND the trade policy
    cfg = dict(cfg or {})
    cfg.setdefault("hold_bucket", DEFAULT_HOLD_BUCKET)
    if kill:
        weights, diagnostics = {}, {"kill_switch": True}
        gated_signals: dict = {}
    else:
        members = store.universe_as_of(as_of)
        symbols = list(members["symbol"])
        # Venue reconciliation BEFORE ranking: a name the broker cannot trade must not
        # receive weight. Fails open (keeps every symbol) if the asset list is unreachable,
        # and says so in the record.
        try:
            symbols, untradable = broker.filter_tradable(symbols)
        except Exception as e:  # noqa: BLE001  asset-list outage must not stop a rebalance
            untradable = {"_error": f"filter_tradable failed, kept all symbols: {e!r}"}
        if untradable:
            print(f"  ! {len(untradable)} universe name(s) not tradable on venue: "
                  f"{', '.join(sorted(k for k in untradable if not k.startswith('_')))}")
        live_names = [p.name for p in live_providers]
        try:
            ic_hist = load_ic_history()
            weights_used, weights_diag = (
                blended_ic_weights(ic_hist, as_of, live_names,
                                   short_window=IC_SHORT_WINDOW, long_window=IC_LONG_WINDOW,
                                   shrink=IC_SHRINK)
                if not ic_hist.empty else (None, {"reason": "no_ic_history"}))
        except Exception as e:  # noqa: BLE001  adaptive weights never block trading
            weights_used, weights_diag = None, {"reason": f"error: {e!r}"}
        ranked = rank_universe(store, live_providers, symbols, as_of,
                               weights=weights_used)
        weights, diagnostics = build_target_portfolio(
            ranked, store, as_of, equity, regime_on=regime_on, cfg=cfg, prev_w=held_w)
        gated_signals = {}
        for p in default_providers():
            if p.name not in gated:
                continue
            try:
                s = p.compute(store, symbols, as_of).dropna()
                gated_signals[p.name] = {k: round(float(v), 4) for k, v in s.items()}
            except Exception as e:  # noqa: BLE001  a gated signal never blocks trading
                gated_signals[p.name] = {"_error": repr(e)}

    # Turnover controls: blend from what we hold toward the sized target, then drop the
    # trades too small to be worth their spread. The backtest applies this exact function
    # to the exact same target, so what was validated is what gets sent.
    #
    # The kill switch is the one thing that must NOT be softened: it liquidates, and a
    # partial adjustment would leave the book part-on in the exact situation the switch
    # exists to exit. Bypass the policy entirely when it fires.
    policy_diag: dict = {}
    target_weights = dict(weights)
    if kill:
        traded_weights = weights          # {} -> full liquidation
        policy_diag = {"bypassed": "kill_switch"}
    else:
        pol = TradePolicy().as_kwargs() if policy is None else dict(policy)
        traded_weights, policy_diag = apply_trade_policy(held_w, target_weights, **pol)

    orders = broker.rebalance(
        TargetPortfolio(traded_weights), dry_run=not submit, band_w=band_w
    )

    record = {
        "as_of": as_of.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "broker": type(broker).__name__,
        "submitted": bool(submit),
        "cadence": cadence,
        "kill_switch": bool(kill),
        "equity": float(equity),
        "equity_source": equity_source,
        # `weights` stays the book we actually hold (post-policy) -- it is what every
        # downstream reader means by "the portfolio". The cost-free ideal is kept
        # alongside it so the gap the turnover controls opened is auditable.
        "weights": traded_weights,
        "target_weights": target_weights,
        "trade_policy": policy_diag,
        "n_orders": len(orders),
        "n_order_errors": sum(1 for o in orders if o.get("status") == "error"),
        "n_order_skips": sum(1 for o in orders if o.get("status") == "skipped"),
        "orders": orders,
        "diagnostics": diagnostics,
        "live_providers": [p.name for p in live_providers],
        "provider_weights": ({k: round(v, 5) for k, v in weights_used.items()}
                             if weights_used else None),
        "weights_diagnostics": weights_diag,
        "gated_signals": gated_signals,
        "untradable": untradable,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{as_of.isoformat()}.json").write_text(
        json.dumps(record, indent=2, default=str))
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="qts weekly rebalance")
    ap.add_argument("--submit", action="store_true",
                    help="actually submit orders (paper unless broker configured live)")
    ap.add_argument("--as-of", type=date.fromisoformat, default=None)
    ap.add_argument("--equity", type=float, default=100_000.0,
                    help="research equity when no broker account is reachable")
    ap.add_argument("--cadence", choices=("weekly", "daily"), default="weekly")
    a = ap.parse_args()
    as_of = a.as_of or date.today()
    # Turnover control now lives in TradePolicy (band + partial adjustment + dust),
    # applied to BOTH cadences and validated in the backtest. compute_orders' own
    # band_w stays at 0 so the band is applied in exactly one place.
    try:
        broker = pick_broker()
        with PITStore(read_only=True) as store:
            rec = run_rebalance(store, as_of, broker, submit=a.submit,
                                equity_fallback=a.equity, cadence=a.cadence)
        print(json.dumps({k: rec[k] for k in
                          ("as_of", "broker", "submitted", "kill_switch",
                           "equity", "n_orders")}, indent=2))
    except Exception as e:  # noqa: BLE001  any failure gets an audit record, never silent
        fail_rec = {
            "as_of": as_of.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error": repr(e),
            "submitted": False,
        }
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        (DEFAULT_OUT_DIR / f"{as_of.isoformat()}-failed.json").write_text(
            json.dumps(fail_rec, indent=2, default=str))
        print(f"! rebalance failed: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
