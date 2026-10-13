"""Symbol normalisation. This looks trivial; it is a live liquidation footgun.

You SUBMIT `BTC/USD` but the position comes back as `BTCUSD`, and DELETE /v2/positions/{sym}
only accepts the slashless form. A book that fails to match its position to its target sees
target 0 and sells. All three forms are pinned here.
"""
import pytest

from qts.crypto.universe import UNIVERSE, from_venue, is_crypto, to_venue


def test_round_trip_both_directions():
    for s in UNIVERSE:
        assert from_venue(to_venue(s)) == s
    assert to_venue("BTC/USD") == "BTCUSD"
    assert from_venue("BTCUSD") == "BTC/USD"


def test_from_venue_is_idempotent_on_canonical_input():
    # The live code path calls this on whatever the API returned; it must not corrupt an
    # already-canonical symbol.
    assert from_venue("ETH/USD") == "ETH/USD"


def test_from_venue_refuses_to_guess():
    # A symbol we cannot map is a symbol we must not trade. Raising beats guessing.
    with pytest.raises(ValueError):
        from_venue("WEIRD")


def test_is_crypto_is_a_whitelist_not_a_pattern():
    for s in UNIVERSE:
        assert is_crypto(s) and is_crypto(to_venue(s))
    # The bug this pins: a naive `endswith("USD")` would classify these as crypto. The
    # equity book uses `not is_crypto()` to decide what it may hold, and a symbol missing
    # from its targets has an IMPLIED TARGET OF ZERO -- i.e. it gets liquidated.
    for s in ("AAPL", "MSFT", "USD", "USDU", "NASDAQ"):
        assert not is_crypto(s), f"{s} must not be treated as crypto"
