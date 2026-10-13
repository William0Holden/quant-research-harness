"""Trade policy: partial adjustment, dust exits, no-trade band."""
from __future__ import annotations

import pytest

from qts.risk.trade_policy import apply_trade_policy


def test_full_adjustment_no_band_reproduces_target():
    """lam=1, no band, no dust == the old free-trading behaviour. This is the guard that
    turning the feature OFF cannot change what we trade."""
    prev = {"AAA": 0.10, "BBB": 0.05}
    target = {"AAA": 0.04, "CCC": 0.08}
    traded, diag = apply_trade_policy(prev, target)
    assert traded == pytest.approx(target)
    assert diag["n_banded"] == 0


def test_partial_adjustment_moves_halfway():
    traded, _ = apply_trade_policy({"AAA": 0.10}, {"AAA": 0.20}, lam=0.5)
    assert traded["AAA"] == pytest.approx(0.15)


def test_partial_adjustment_on_a_new_name_opens_partially():
    traded, _ = apply_trade_policy({}, {"AAA": 0.10}, lam=0.5)
    assert traded["AAA"] == pytest.approx(0.05)


def test_band_suppresses_small_drift_but_keeps_the_position():
    """A 20bp drift inside a 50bp band must leave the position untouched, not drop it."""
    traded, diag = apply_trade_policy({"AAA": 0.050}, {"AAA": 0.052}, band_w=0.005)
    assert traded["AAA"] == pytest.approx(0.050)   # held at PREV, not moved to target
    assert diag["n_banded"] == 1
    assert diag["turnover"] == pytest.approx(0.0)


def test_band_allows_a_move_larger_than_the_band():
    traded, diag = apply_trade_policy({"AAA": 0.05}, {"AAA": 0.08}, band_w=0.005)
    assert traded["AAA"] == pytest.approx(0.08)
    assert diag["n_banded"] == 0


def test_full_exit_always_trades_even_inside_the_band():
    """The band is a drift filter, not a risk filter: an exit is never suppressed."""
    traded, diag = apply_trade_policy({"AAA": 0.001}, {}, band_w=0.005)
    assert "AAA" not in traded
    assert diag["turnover"] == pytest.approx(0.001)


def test_dust_rule_closes_a_decayed_position_rather_than_carrying_it():
    """Without this, lam<1 leaves stubs decaying toward zero forever, paying spread."""
    traded, diag = apply_trade_policy({"AAA": 0.003}, {}, lam=0.5, dust_w=0.0025)
    assert "AAA" not in traded          # 0.003 -> 0.0015 -> under dust -> closed
    assert diag["n_dust_exits"] == 1


def test_dust_rule_does_not_open_a_position_too_small_to_matter():
    traded, _ = apply_trade_policy({}, {"AAA": 0.001}, dust_w=0.0025)
    assert traded == {}


def test_band_and_partial_adjustment_compose():
    """lam shrinks the move; the band then judges the SHRUNKEN move, not the raw one.
    0.10 -> target 0.13 with lam=0.1 is a 30bp move, inside a 50bp band -> no trade."""
    traded, diag = apply_trade_policy({"AAA": 0.10}, {"AAA": 0.13}, lam=0.1, band_w=0.005)
    assert traded["AAA"] == pytest.approx(0.10)
    assert diag["n_banded"] == 1


def test_turnover_and_gross_reported():
    traded, diag = apply_trade_policy({"AAA": 0.10}, {"AAA": 0.05, "BBB": 0.05})
    assert diag["turnover"] == pytest.approx(0.10)   # 0.05 out of AAA + 0.05 into BBB
    assert diag["gross"] == pytest.approx(0.10)
    assert diag["target_gross"] == pytest.approx(0.10)


def test_liquidation_target_empties_the_book():
    traded, _ = apply_trade_policy({"AAA": 0.10, "BBB": 0.05}, {}, band_w=0.005, lam=1.0)
    assert traded == {}


@pytest.mark.parametrize("lam", [0.0, -0.1, 1.5])
def test_invalid_lambda_raises(lam):
    """A silent clamp would quietly change the strategy; this value comes from config."""
    with pytest.raises(ValueError):
        apply_trade_policy({}, {"AAA": 0.1}, lam=lam)


# --- gross guard -----------------------------------------------------------------
# Cash earns nothing, so a rising book drifts UP as a share of equity while the band
# suppresses the trim that would correct it. Unchecked, that ratchet turned the sweep's
# top configs into books running at 1.28x their intended gross -- extra beta masquerading
# as alpha. The guard bounds it.

def test_gross_guard_trims_a_book_that_drifted_hot():
    prev = {"AAA": 0.60, "BBB": 0.60}          # drifted far above target
    target = {"AAA": 0.35, "BBB": 0.35}        # sizer wants 0.70 gross
    traded, diag = apply_trade_policy(prev, target, band_w=0.05, lam=0.1,
                                      max_gross_drift=0.05)
    assert diag["gross_scaled_back"] is True
    assert diag["gross"] == pytest.approx(0.70 * 1.05)      # trimmed to the cap
    assert traded["AAA"] == pytest.approx(traded["BBB"])    # trim is proportional


def test_gross_guard_leaves_a_normal_book_alone():
    """The guard must be rare. Ordinary drift inside tolerance must not cause churn."""
    traded, diag = apply_trade_policy({"AAA": 0.10}, {"AAA": 0.10}, band_w=0.005,
                                      max_gross_drift=0.05)
    assert diag["gross_scaled_back"] is False
    assert traded["AAA"] == pytest.approx(0.10)


def test_gross_guard_never_levers_a_book_UP():
    """It is a ceiling, not a target: an under-invested book is not scaled up to meet it
    (that would override the band and the partial adjustment)."""
    traded, diag = apply_trade_policy({"AAA": 0.01}, {"AAA": 0.50}, lam=0.1,
                                      max_gross_drift=0.05)
    assert diag["gross_scaled_back"] is False
    assert traded["AAA"] == pytest.approx(0.01 + 0.1 * 0.49)


def test_gross_guard_ignores_an_empty_target():
    """Liquidation: target_gross is 0, so there is no cap to scale toward -- and dividing
    by it would be a crash in the one code path that must never fail."""
    traded, diag = apply_trade_policy({"AAA": 0.10}, {}, max_gross_drift=0.05)
    assert traded == {}
    assert diag["gross_scaled_back"] is False


def test_gross_guard_can_be_disabled_with_none():
    """max_gross_drift=None disables the guard. It must not crash: this is the exact
    signature used to reproduce the pre-guard (incumbent) behaviour for A/B comparison,
    and it did crash the first time it was run."""
    traded, diag = apply_trade_policy({"AAA": 0.9}, {"AAA": 0.1}, lam=0.1,
                                      max_gross_drift=None)
    assert diag["gross_scaled_back"] is False
    assert traded["AAA"] == pytest.approx(0.9 - 0.1 * 0.8)
