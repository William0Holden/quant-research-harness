"""Turnover-aware trade policy: cost-free target weights -> weights we actually hold.

`build_target_portfolio` answers "what would I hold if trading were free?". Trading is
not free, so this module answers the second question: "given what I already hold, and
what it costs to move, what should I hold?"

Three levers, applied in this order:

  1. PARTIAL ADJUSTMENT (lam): move only part of the way to the target.
        desired = prev + lam * (target - prev)
     Gârleanu-Pedersen: with linear costs and slowly-decaying alpha, snapping all the
     way to the target every period is optimal only when trading is free. lam=1.0 is
     the old (free-trading) behaviour.

  2. DUST RULE (dust_w): a position that partial adjustment has ground down below
     dust_w -- or a new position that would open below it -- is snapped to zero.
     Without this, lam<1 leaves an ever-growing tail of decaying stubs the book pays
     spread on forever.

  3. NO-TRADE BAND (band_w): if a name's weight would move by less than band_w, leave
     it where it is. Full exits (desired == 0 while held) ALWAYS trade -- an exit is a
     risk decision, not a drift correction.

Deliberately NO renormalisation at the end. Rescaling the book to hit a gross target
would perturb every held name by a few bps and hand those trades straight back to the
band, defeating the whole point. Gross is therefore allowed to drift a little below the
sized target; the cash remainder absorbs it, and the per-position caps upstream still bind.

Pure: no store, no network, no clock. Both `backtest.engine` and `exec.rebalance` call
this same function, so the simulated book and the traded book cannot drift apart.
"""
from __future__ import annotations

# Defaults are deliberately CONSERVATIVE (= old behaviour) so that importing this
# module changes nothing until a caller opts in. Live values live in config.TradePolicy.
NO_BAND = 0.0
FULL_ADJUSTMENT = 1.0
NO_DUST = 0.0

# How far the held book may drift above the gross the sizer asked for before it is
# trimmed back. 5% of the target gross -- tight enough that "cost saving" can never
# turn into unintended leverage, loose enough that ordinary drift doesn't cause churn.
DEFAULT_MAX_GROSS_DRIFT = 0.05


def apply_trade_policy(
    prev_w: dict[str, float],
    target_w: dict[str, float],
    band_w: float = NO_BAND,
    lam: float = FULL_ADJUSTMENT,
    dust_w: float = NO_DUST,
    max_gross_drift: float = DEFAULT_MAX_GROSS_DRIFT,
) -> tuple[dict[str, float], dict]:
    """Blend current holdings toward the target, then suppress trades not worth making.

    Args:
        prev_w:   symbol -> weight currently held (market value / equity).
        target_w: symbol -> cost-free target weight from the sizing pipeline.
        band_w:   no-trade band in weight units (0.005 = 50bps of equity).
        lam:      partial-adjustment fraction in (0, 1]. 1.0 = snap fully to target.
        dust_w:   positions below this weight are closed rather than carried.

    Returns:
        (traded_w, diagnostics). traded_w is what to hold: pass it to compute_orders as
        the target. Names at exactly 0 are dropped from the dict.

    Raises:
        ValueError: if lam is outside (0, 1] -- a silent clamp here would quietly change
            the strategy, and this value comes from config/CLI where a typo is plausible.
    """
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"lam must be in (0, 1], got {lam}")
    if band_w < 0 or dust_w < 0:
        raise ValueError(f"band_w and dust_w must be non-negative, got {band_w}, {dust_w}")

    names = set(prev_w) | set(target_w)
    traded: dict[str, float] = {}
    n_banded = n_dust = 0

    for sym in names:
        prev = float(prev_w.get(sym, 0.0))
        target = float(target_w.get(sym, 0.0))

        # 1. partial adjustment
        desired = prev + lam * (target - prev)

        # 2. dust: don't carry (or open) a position too small to matter. A name the
        #    sizer still wants is only dusted if it is genuinely tiny; a name the sizer
        #    has dropped (target == 0) is closed as soon as lam has ground it under the
        #    threshold, rather than decaying toward zero over many paid rebalances.
        if desired < dust_w:
            desired = 0.0
            if prev > 0.0:
                n_dust += 1

        # 3. no-trade band -- but never suppress a full exit.
        is_full_exit = desired == 0.0 and prev > 0.0
        if band_w and abs(desired - prev) < band_w and not is_full_exit:
            desired = prev
            n_banded += 1

        if desired > 0.0:
            traded[sym] = desired

    # 4. GROSS GUARD. Cash earns nothing, so when the book rises the held weights drift
    #    UP as a share of equity -- and steps 1-3 are all designed to suppress trades,
    #    including the trim that would bring them back. Left alone this ratchets: over a
    #    bull run the book ends up materially more exposed than the vol target asked for
    #    (the 2022-2026 sweep produced books at 1.28x their intended gross, whose extra
    #    "alpha" was just extra beta). Cost control may not quietly become leverage, so
    #    once the drift exceeds the tolerance we scale back to it. Rare by construction,
    #    and a scale-down is the cheapest possible way to shed exposure.
    target_gross = sum(target_w.values())
    gross = sum(traded.values())
    scaled_back = False
    if max_gross_drift is not None and target_gross > 0:
        cap = target_gross * (1.0 + max_gross_drift)
        if gross > cap:
            traded = {s: w * (cap / gross) for s, w in traded.items()}
            scaled_back = True

    turnover = sum(
        abs(traded.get(s, 0.0) - float(prev_w.get(s, 0.0)))
        for s in set(traded) | set(prev_w)
    )
    diag = {
        "band_w": band_w,
        "lam": lam,
        "dust_w": dust_w,
        "n_prev": len(prev_w),
        "n_target": len(target_w),
        "n_held": len(traded),
        "n_banded": n_banded,          # names left alone because the move was too small
        "n_dust_exits": n_dust,        # names closed for being below dust_w
        "turnover": turnover,          # sum |dw|, in equity units (1.0 = whole book)
        "gross": sum(traded.values()),
        "target_gross": target_gross,
        "gross_scaled_back": scaled_back,   # the ratchet was caught and trimmed
    }
    return traded, diag
