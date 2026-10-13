"""Asset-class profile: the ONE object that keeps the two books from eating each other.

THE FOOTGUN THIS EXISTS TO DISARM (verified against the live account 2026-07-14):
Alpaca paper account PA3QQCS34EZG has crypto_status=ACTIVE and holds 101 us_equity
positions. Crypto and equities live in the SAME account. `AlpacaBroker.get_positions()`
returns EVERY position as market_value/equity. So the moment a crypto book exists in that
account, the Friday equity rebalance sees BTCUSD, finds it absent from its target weights
(= implied target 0), and CLOSES IT. And vice versa. Each book silently liquidates the
other, every week, and the kill switch blends both books into one drawdown number.

The fix is not a filter -- a filter is one missed `if` away from a liquidation. The fix is
a SECOND PAPER ACCOUNT. The crypto profile resolves different credentials, a different
records dir and a different kill switch, so the crypto book cannot see, price, or trade an
equity position even in principle. Blast radius: zero.

`tests/test_crypto_profile_seam.py` pins this.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ..config import OUTPUTS_DIR
from .costs import CryptoCostModel
from .universe import UNIVERSE


@dataclass(frozen=True)
class CryptoProfile:
    """Everything that differs from the equity book. Nothing else may differ implicitly."""

    symbols: tuple[str, ...] = UNIVERSE
    cost_model: CryptoCostModel = field(default_factory=CryptoCostModel)

    # Sizing. See sizing.py: Sharpe is FLAT across vol targets (0.42 -> 0.54 measured), so
    # this is a risk dial, not an alpha lever. 40% lands realized book vol near 25%.
    vol_target: float = float(os.environ.get("QTS_CRYPTO_VOL_TARGET", "0.40"))
    max_gross: float = 1.0                       # never lever a long-only coin book

    # No-trade band. Chosen from the COST RATIO, not from the backtest sweep's winner: the
    # sweep's Sharpe spread (0.52-0.57) is well inside its own standard error of 0.47, so
    # picking the best-looking band would be fitting noise. The COST saving is the certain
    # gain (3.19% -> ~2.0%/yr), exactly as on the equity book. Garleanu-Pedersen: the
    # optimal band widens with cost, and crypto costs 4-7x equities.
    band_w: float = float(os.environ.get("QTS_CRYPTO_BAND_W", "0.05"))

    # Kill switch. RATIO DISCIPLINE: the equity book is 25% DD / 10% vol = 2.5x. Crypto at
    # 50% / 20% = 2.5x -- consistent. The originally proposed 50%/40% = 1.25x sat BELOW the
    # historically realized crypto DD/vol ratio of 1.3-1.5x, so it would have fired in every
    # bear market: that is a bear-market participation rule wearing a kill switch's name.
    kill_switch_dd: float = float(os.environ.get("QTS_CRYPTO_KILL_DD", "0.50"))

    # Sentiment overlay. OFF until it clears the falsification bar in the design doc
    # (orthogonalise vs lagged returns/vol/turnover; ablate against price-only; bot-filter).
    overlay_on: bool = os.environ.get("QTS_CRYPTO_OVERLAY", "0") == "1"

    # Execution window: Tue-Thu 14:00-20:00 UTC (US-hours liquidity). Deliberately NOT the
    # equity book's Friday 23:30 UTC -- in crypto that is Friday night, the thinnest window
    # of the week (weekend volume -20-25%, costs +11%).
    exec_days: tuple[int, ...] = (1, 2, 3)       # Mon=0 .. Tue/Wed/Thu
    exec_hours_utc: tuple[int, int] = (14, 20)

    # Circuit breaker (Oct-10-2025 insurance: depth fell >90%, spreads went to double-digit
    # PERCENT). Above this spread we SKIP the rebalance rather than trade through it.
    max_half_spread_bps: float = float(os.environ.get("QTS_CRYPTO_MAX_SPREAD_BPS", "50"))

    # Dust floor for a single order. Inside the no-trade band the held weight is left
    # alone, but the QUANTITY still drifts with price, and the old $1 floor let that drift
    # go out as $7-$10 orders (0.00008848 BTC on 2026-08-26) that paid a spread to move
    # nothing. Full exits are exempt -- an exit is a risk decision, not drift.
    min_order_usd: float = float(os.environ.get("QTS_CRYPTO_MIN_ORDER_USD", "25"))

    @property
    def records_dir(self) -> Path:
        return Path(os.environ.get("QTS_CRYPTO_RECORDS",
                                   OUTPUTS_DIR / "rebalances_crypto"))

    @property
    def ledger_path(self) -> Path:
        """Lifetime fill ledger (JSONL). Lives beside the records, not inside them, so
        archiving the records dir to reset the kill switch does not erase the cost history."""
        return Path(os.environ.get("QTS_CRYPTO_LEDGER", OUTPUTS_DIR / "crypto_fills.jsonl"))

    def broker_keys(self) -> tuple[str | None, str | None]:
        """SEPARATE credentials. Falls back to the equity keys ONLY if the crypto-specific
        ones are absent -- and `guard_account()` refuses to run in that case, because
        sharing an account with the equity book is precisely the failure this class exists
        to prevent."""
        return (os.environ.get("APCA_CRYPTO_KEY_ID"),
                os.environ.get("APCA_CRYPTO_SECRET_KEY"))

    def has_own_account(self) -> bool:
        """True only if the crypto keys exist AND are genuinely a DIFFERENT account.

        Checking only that the keys exist is not enough: pointing APCA_CRYPTO_KEY_ID at the
        equity key would sail through such a check and put both books back in one account --
        which is the entire failure this class exists to prevent, restored by a copy-paste.
        """
        k, sec = self.broker_keys()
        if not (k and sec):
            return False
        eq_k = os.environ.get("APCA_API_KEY_ID")
        return not (eq_k and k == eq_k)

    def guard_account(self) -> None:
        """Refuse to trade crypto out of the equity account. Raises, loudly.

        A soft warning here would be worthless: the failure is silent (a liquidated book)
        and only visible after the fact.
        """
        k, _ = self.broker_keys()
        if k and k == os.environ.get("APCA_API_KEY_ID"):
            raise RuntimeError(
                "APCA_CRYPTO_KEY_ID is the SAME key as APCA_API_KEY_ID -- the crypto book "
                "would trade the equity account. Crypto and equities share one account at "
                "Alpaca, so each book would liquidate the other. Open a SECOND paper account "
                "and use its keys."
            )
        if not self.has_own_account():
            raise RuntimeError(
                "Crypto book has no dedicated Alpaca account. Set APCA_CRYPTO_KEY_ID and "
                "APCA_CRYPTO_SECRET_KEY from a SECOND paper account "
                "(app.alpaca.markets -> click the paper account number, upper-left -> "
                "'Open New Paper Account').\n\n"
                "Refusing to run against the equity account: crypto and equities share one "
                "account at Alpaca, so the equity rebalance would see BTCUSD at an implied "
                "target weight of 0 and liquidate it -- and the crypto rebalance would do "
                "the same to your 101 equity positions."
            )
