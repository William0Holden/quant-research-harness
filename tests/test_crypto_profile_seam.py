"""THE SEAM. The equity and crypto books must be incapable of touching each other.

The real footgun (verified against the live account): Alpaca paper PA3QQCS34EZG has
crypto_status=ACTIVE and holds 101 us_equity positions. Crypto and equities share ONE
account. AlpacaBroker.get_positions() returns EVERY position as market_value/equity, and a
symbol absent from a target dict has an IMPLIED TARGET OF ZERO. So a crypto book in that
account means: the Friday equity rebalance sees BTCUSD, targets it at 0, and LIQUIDATES IT.
And the crypto rebalance does the same to 101 equity positions.

The defence is not a filter (one missed `if` away from a liquidation) -- it is a separate
account, enforced by a guard that RAISES.
"""
import json

import pytest

from qts.crypto.broker import CryptoBroker
from qts.crypto.profile import CryptoProfile


def test_guard_refuses_to_run_without_a_dedicated_account(monkeypatch):
    monkeypatch.delenv("APCA_CRYPTO_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_CRYPTO_SECRET_KEY", raising=False)
    # Equity keys present -- the dangerous case: everything would "work", against the wrong book.
    monkeypatch.setenv("APCA_API_KEY_ID", "equity-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "equity-secret")

    p = CryptoProfile()
    assert not p.has_own_account()
    with pytest.raises(RuntimeError, match="no dedicated Alpaca account"):
        p.guard_account()


def test_guard_passes_with_dedicated_keys(monkeypatch):
    monkeypatch.setenv("APCA_CRYPTO_KEY_ID", "crypto-key")
    monkeypatch.setenv("APCA_CRYPTO_SECRET_KEY", "crypto-secret")
    p = CryptoProfile()
    assert p.has_own_account()
    p.guard_account()      # must not raise


def test_crypto_broker_never_reads_the_equity_keys(monkeypatch):
    """If it fell back to APCA_API_KEY_ID it would happily trade the equity account."""
    monkeypatch.setenv("APCA_API_KEY_ID", "equity-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "equity-secret")
    monkeypatch.delenv("APCA_CRYPTO_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_CRYPTO_SECRET_KEY", raising=False)
    b = CryptoBroker()
    with pytest.raises(RuntimeError, match="no credentials"):
        _ = b._h


def test_broker_refuses_an_account_holding_equities(monkeypatch):
    """Belt and braces: even if the guard were bypassed and the crypto book were pointed at
    the equity account, get_positions() must REFUSE rather than price 101 equity positions
    into a 2-asset crypto book (where they would all be targeted at zero, i.e. sold)."""
    b = CryptoBroker(key_id="k", secret_key="s")
    monkeypatch.setattr(b, "equity", lambda: 100_000.0)
    monkeypatch.setattr(b, "_get", lambda path, **kw: [
        {"symbol": "AAPL", "asset_class": "us_equity", "market_value": "5000"},
    ])
    with pytest.raises(RuntimeError, match="pointed at the equity account"):
        b.get_positions()


def test_records_dir_is_separate_from_the_equity_book():
    p = CryptoProfile()
    assert p.records_dir.name == "rebalances_crypto"
    assert p.records_dir.name != "rebalances"


def test_kill_switch_ratio_is_consistent_with_the_equity_book():
    """Ratio discipline. Equity: 25% DD / 10% vol = 2.5x -- a genuine tail breaker.
    The originally proposed crypto 50% / 40% = 1.25x sits BELOW the historically realized
    crypto DD/vol of 1.3-1.5x, so it would fire in EVERY bear market. That is a bear-market
    participation rule wearing a kill switch's name, and it is the sort of thing that exits
    at -50% with no rule for getting back in -- while the median post-70%-decline BTC rally
    is +1,692%."""
    p = CryptoProfile()
    # Realized book vol lands ~25% (trend exposure averages ~0.5 and multiplies the vol
    # scalar DOWN), so the operative ratio is 0.50 / 0.25 = 2.0x.
    realized_vol_est = p.vol_target * 0.6
    ratio = p.kill_switch_dd / realized_vol_est
    assert ratio >= 1.6, f"kill switch would fire in an ordinary bear market (ratio {ratio:.2f})"


def test_guard_rejects_crypto_keys_that_are_actually_the_equity_keys(monkeypatch):
    """The hole in the first version of this guard. Checking only that the crypto keys EXIST
    lets a copy-paste put both books back in one account -- and everything would appear to
    work, right up until each book liquidated the other."""
    monkeypatch.setenv("APCA_API_KEY_ID", "SAME-KEY")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "equity-secret")
    monkeypatch.setenv("APCA_CRYPTO_KEY_ID", "SAME-KEY")
    monkeypatch.setenv("APCA_CRYPTO_SECRET_KEY", "whatever")

    p = CryptoProfile()
    assert not p.has_own_account()
    with pytest.raises(RuntimeError, match="SAME key"):
        p.guard_account()


def test_equity_broker_ignores_crypto_rather_than_liquidating_it(monkeypatch):
    """Belt and braces on the OTHER side. If crypto ever lands in the equity account, the
    equity book must skip it -- not target it at zero and sell it."""
    from qts.exec.broker import AlpacaBroker

    class _P:
        def __init__(self, symbol, asset_class, mv):
            self.symbol, self.asset_class, self.market_value = symbol, asset_class, mv

    class _Client:
        def get_account(self):
            return type("A", (), {"equity": "100000"})()
        def get_all_positions(self):
            return [_P("AAPL", "us_equity", "5000"), _P("BTCUSD", "crypto", "9000")]

    b = AlpacaBroker(key_id="k", secret_key="s")
    monkeypatch.setattr(b, "_trading_client", lambda: _Client())
    w = b.get_positions()
    assert "AAPL" in w
    assert "BTCUSD" not in w, "equity book must NOT see crypto -- it would sell it"
