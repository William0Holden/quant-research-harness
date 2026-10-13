"""Live crypto rebalance. Trend ensemble -> sizer -> trade policy -> maker ladder.

Every run writes an audited JSON record to outputs/rebalances_crypto/<ts>.json containing
BOTH the traded weights and the overlay-off counterfactual, so the sentiment overlay's
contribution stays exactly attributable:

    r_book(t) = m(t-1) * r_trend(t)

and a shadow overlay-off NAV can be reconstructed from the record alone. If the overlay ever
makes or loses money, we will know it was the overlay and not the trend.

The record also carries the COST LEDGER, because the paper account charges no crypto fees --
so the broker's reported equity is not the truth and `adjusted_equity` is.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from ..risk.trade_policy import apply_trade_policy
from .broker import CryptoBroker, VenueRejected
from .prices import fetch_panel, latest_quote
from .profile import CryptoProfile
from .sizing import build_crypto_targets
from .trend import exposure_as_of
from .universe import UNIVERSE


def sentiment_z(profile: CryptoProfile) -> float | None:
    """Aggregate crypto sentiment z-score, or None.

    None is the GPU-DOWN PATH and it is load-bearing: the GPU worker is shared with the user, so
    when it is busy or asleep the score goes stale, this returns None, the overlay
    multiplier collapses to 1.0, and the book trades on TREND ALONE. Degrade to the
    price-only strategy -- never block, never crash, never fight the user for the machine.

    Wired to a real provider only once the overlay clears the falsification bar in the
    design doc (orthogonalise vs lagged returns/vol/turnover; ablate against price-only;
    bot-filter). Until then it is honestly None and the overlay is off.
    """
    return None


def plan(profile: CryptoProfile, broker: CryptoBroker | None = None) -> dict:
    panel = fetch_panel(profile.symbols)
    as_of = panel.index[-1]

    te = {s: exposure_as_of(panel[s], as_of) for s in profile.symbols}
    closes = {s: panel[s] for s in profile.symbols}
    z = sentiment_z(profile) if profile.overlay_on else None

    target, diag = build_crypto_targets(
        te, closes, sent_z=z, vol_target=profile.vol_target,
        max_gross=profile.max_gross, overlay_on=profile.overlay_on,
    )
    # The overlay-off counterfactual, recorded every run -- this is what makes attribution
    # exact rather than a story we tell afterwards.
    trend_only, _ = build_crypto_targets(
        te, closes, sent_z=None, vol_target=profile.vol_target,
        max_gross=profile.max_gross, overlay_on=False,
    )

    prev = broker.get_positions() if broker else {}
    traded, pol = apply_trade_policy(prev, target, band_w=profile.band_w, lam=1.0,
                                     dust_w=0.0, max_gross_drift=None)

    return {
        "as_of": str(as_of.date()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "prev_weights": prev,
        "target_weights": target,        # what the sizer asked for
        "trend_weights": trend_only,     # overlay-OFF counterfactual
        "weights": traded,               # what we will actually hold, post-band
        "sizing": diag,
        "trade_policy": pol,
        "sentiment": _sentiment_snapshot(profile, as_of.date(), closes),
        "profile": {"vol_target": profile.vol_target, "band_w": profile.band_w,
                    "overlay_on": profile.overlay_on,
                    "kill_switch_dd": profile.kill_switch_dd},
    }


def _sentiment_snapshot(profile: CryptoProfile, as_of, closes: dict) -> dict | None:
    """Compute the three GATED sentiment risk signals for the record. Best-effort and
    strictly read-only: it affects NOTHING (sizing, overlay, execution). If the PIT store or
    the data is unavailable it returns None -- a sentiment failure must never block a trade.

    Recorded every run so the forward risk-IC record accrues (qts.crypto.sentiment.research);
    each signal earns un-gating only on that record, by a later deliberate decision."""
    try:
        from ..pit.store import PITStore
        from .sentiment.providers import CryptoSentimentSignals
        from .sizing import realized_vol
        vols = {s: realized_vol(closes[s]) for s in closes}
        with PITStore(read_only=True) as store:
            return CryptoSentimentSignals(store).compute_all(
                list(profile.symbols), as_of, vols)
    except Exception as e:  # noqa: BLE001  a gated, log-only signal may never break trading
        return {"error": str(e), "gated": True}


def main() -> None:
    ap = argparse.ArgumentParser(description="crypto trend book rebalance")
    ap.add_argument("--live", action="store_true", help="actually send orders")
    ap.add_argument("--passive-timeout", type=int, default=None,
                    help="seconds to rest passively before escalating to a marketable IOC. "
                         "Default (None) uses broker.PASSIVE_TIMEOUT_S. Lower it for a "
                         "watched first run; our alpha does not decay in minutes.")
    a = ap.parse_args()

    profile = CryptoProfile()
    profile.guard_account()          # refuses to trade crypto out of the equity account

    broker = CryptoBroker(max_half_spread_bps=profile.max_half_spread_bps,
                          ledger_path=profile.ledger_path)
    rec = plan(profile, broker)
    equity = broker.equity()

    # Circuit breaker BEFORE any order: if the book is broken, skip the whole rebalance.
    quotes = {}
    for s in profile.symbols:
        ok, q = broker.tradeable(s)
        quotes[s] = {**{k: round(v, 6) for k, v in q.items()}, "tradeable": ok}
    rec["quotes"] = quotes
    if not all(v["tradeable"] for v in quotes.values()):
        rec["status"] = "SKIPPED_WIDE_SPREAD"
        _write(profile, rec)
        print("SKIPPED: spread above circuit-breaker threshold. No orders sent.")
        return

    orders = []
    for s in profile.symbols:
        tgt_w = rec["weights"].get(s, 0.0)
        px = quotes[s]["mid"]
        want_qty = tgt_w * equity / px
        have_qty = broker.get_qty(s) if a.live else 0.0
        d = want_qty - have_qty
        # A full exit always trades (it is a risk decision); anything else below the
        # dust floor is price drift inside the no-trade band, not a decision, and a
        # $7 sell of 0.00008848 BTC pays a spread to accomplish nothing.
        is_full_exit = tgt_w == 0.0 and have_qty > 0.0
        if abs(d * px) < profile.min_order_usd and not is_full_exit:
            orders.append({"symbol": s, "side": "buy" if d > 0 else "sell", "qty": abs(d),
                           "status": "skipped_dust", "notional_usd": round(abs(d * px), 2)})
            continue
        kw = {} if a.passive_timeout is None else {"passive_timeout_s": a.passive_timeout}
        try:
            orders.append(broker.execute(s, d, dry_run=not a.live, **kw))
        except VenueRejected as e:
            # One symbol's rejection must not strand the other's rebalance -- and the
            # venue's own explanation goes in the record, where it can be read later.
            orders.append({"symbol": s, "side": "buy" if d > 0 else "sell", "qty": abs(d),
                           "status": "error", "http_status": e.status, "error": e.body,
                           "sent": e.payload})
            print(f"  ! {s}: venue rejected order: {e}")

    rec.update(equity=equity, equity_source="broker", orders=orders,
               cost_ledger=broker.ledger.summary(),            # lifetime of the book
               cost_ledger_run=broker.ledger.run_summary(),    # this rebalance only
               adjusted_equity=broker.ledger.adjusted_equity(equity),
               n_order_errors=sum(1 for o in orders if o.get("status") == "error"),
               status="LIVE" if a.live else "DRY_RUN")
    _write(profile, rec)

    print(f"as_of {rec['as_of']}  equity ${equity:,.2f}  gross {rec['sizing']['gross']:.3f} "
          f"(trend {rec['sizing']['trend_gross']:.3f}, overlay m={rec['sizing']['overlay_m']})")
    for s in profile.symbols:
        e = rec['sizing']['trend_exposure'][s]
        print(f"  {s:9s} trend {('--' if e is None else f'{e:.2f}')}  "
              f"-> w {rec['weights'].get(s, 0.0):.4f}")
    for o in orders:
        print(f"  {o['side']:4s} {o['symbol']:9s} qty {o['qty']:.6f}  {o['status']}"
              + (f"  ({o['error'][:120]})" if o.get("error") else ""))
    if any(o.get("status") == "error" for o in orders):
        raise SystemExit(2)      # visible in the cron log and the exit status, after the record is written
    if not orders:
        print("  no orders: book already inside the no-trade band")


def _write(profile: CryptoProfile, rec: dict) -> None:
    profile.records_dir.mkdir(parents=True, exist_ok=True)
    p = profile.records_dir / f"{rec['ts'].replace(':', '-')}.json"
    p.write_text(json.dumps(rec, indent=2, default=str))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
