"""Central config + paths. No secrets here; broker keys come from env at the exec layer."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root = two parents up from this file (src/qts/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("QTS_DATA_DIR", REPO_ROOT / "data"))
OUTPUTS_DIR = Path(os.environ.get("QTS_OUTPUTS_DIR", REPO_ROOT / "outputs"))
DB_PATH = Path(os.environ.get("QTS_DB_PATH", DATA_DIR / "qts.duckdb"))
CORPUS_DB_PATH = Path(os.environ.get("QTS_CORPUS_DB_PATH", DATA_DIR / "corpus.duckdb"))

# SEC requires a descriptive User-Agent with contact info. Override via env in prod.
SEC_USER_AGENT = os.environ.get(
    "QTS_SEC_UA", "qts-research research@example.com"
)
SEC_RATE_SLEEP_S = float(os.environ.get("QTS_SEC_SLEEP", "0.15"))  # ~7 req/s, under SEC's 10


@dataclass(frozen=True)
class CostModel:
    """Mandatory non-zero defaults. Zeroing any of these is a conscious, logged override.

    Leak guard #5: a backtest cannot silently assume frictionless fills.
    """
    commission_bps: float = 0.0      # Alpaca commission-free equities
    spread_bps: float = 5.0          # half-spread paid per side, liquid large/mid cap
    slippage_bps: float = 3.0        # market-impact estimate per side
    borrow_bps_annual: float = 50.0  # short-leg borrow cost (research/paper only)
    st_gains_tax_rate: float = 0.0   # applied in after-tax reporting, not per-trade

    def per_side_bps(self) -> float:
        return self.commission_bps + self.spread_bps + self.slippage_bps


@dataclass(frozen=True)
class TradePolicy:
    """Turnover controls applied between the sized target and the orders we send.

    See risk/trade_policy.py for semantics. Defaults are chosen by
    `qts.research.turnover_study`, which sweeps them through the backtest against the
    CostModel above and reports net-of-cost Sharpe -- do not hand-tune them here without
    re-running it. Env overrides exist so a run can be pinned without a code change.
    """
    # Set by the 2022-2026 sweep (outputs/turnover_study.csv, weekly cadence):
    #   net Sharpe 0.608 vs 0.588 incumbent, cost 0.83%/yr vs 0.97%, fidelity 0.969.
    # The Sharpe difference is inside the noise; the COST saving is the real, certain
    # gain. band_w is 5bp, NOT the 50bp that intuition suggests -- the book holds ~100
    # names at ~0.65% each, so a 50bp band is 3/4 of a whole position and freezes the
    # portfolio into cash (see the design doc).
    band_w: float = float(os.environ.get("QTS_BAND_W", "0.0005"))    # 5bp no-trade band
    lam: float = float(os.environ.get("QTS_LAMBDA", "0.8"))          # partial adjustment
    dust_w: float = float(os.environ.get("QTS_DUST_W", "0.0015"))    # close sub-15bp stubs
    # Ceiling on how far the held book may drift ABOVE the gross the sizer asked for.
    # Cash earns nothing, so a rising book ratchets up while the band suppresses the trim.
    max_gross_drift: float = float(os.environ.get("QTS_MAX_GROSS_DRIFT", "0.05"))

    def as_kwargs(self) -> dict:
        return {"band_w": self.band_w, "lam": self.lam, "dust_w": self.dust_w,
                "max_gross_drift": self.max_gross_drift}


# Rebalance cadence. The sweep's point estimate favours monthly (Sharpe 0.698) over
# weekly (0.588), but the gap is 0.7 sigma -- noise -- and a slow cadence would discard
# the 5-day-half-life sentiment signals this system just built. Weekly stands.
DEFAULT_FREQ = os.environ.get("QTS_FREQ", "W-FRI")

# Membership hysteresis (see risk/sizing.hysteresis_members). A held name is kept while
# it still ranks at or above this quintile bucket (5 = top); it is sold only when it
# falls below. 0 disables. Set by qts.research.churn_study, 2022-01..2026-06 weekly,
# live adaptive weights (outputs/churn_study.csv):
#   hold_bucket=0: turnover 14.8%/rebal, cost 0.62%/yr, net Sharpe 0.899, 101 names
#   hold_bucket=4: turnover  6.4%/rebal, cost 0.26%/yr, net Sharpe 0.850, 135 names
#   hold_bucket=3: turnover  3.9%/rebal, cost 0.16%/yr, net Sharpe 0.626, 168 names (diluted)
# The Sharpe gap at 4 is inside the noise; the cost saving is certain. 3 dilutes the signal.
DEFAULT_HOLD_BUCKET = int(os.environ.get("QTS_HOLD_BUCKET", "4"))

# Adaptive provider-weight smoothing (research/ic.blended_ic_weights). Weekly ICs are
# noise (sd 0.1-0.27 around means near 0.01). churn_study tried 26/52 shrink .75 and
# 52/52 shrink .75: turnover barely moved (17.6-17.5% vs 19.5%/rebal, 2025+) and Sharpe
# fell (0.44 vs 0.63). Smoothing is NOT the churn lever; membership hysteresis is.
IC_SHORT_WINDOW = int(os.environ.get("QTS_IC_SHORT_WINDOW", "13"))
IC_LONG_WINDOW = int(os.environ.get("QTS_IC_LONG_WINDOW", "52"))
IC_SHRINK = float(os.environ.get("QTS_IC_SHRINK", "0.5"))

# Conviction sizing: the sweep found NO benefit (mean net Sharpe 0.636 off vs 0.632 on).
# Implemented and tested, off by default. Turn on only if a future study earns it.
DEFAULT_CONVICTION = os.environ.get("QTS_CONVICTION", "0") == "1"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ---- live IC gate ----------------------------------------------------------
# Providers listed here compute + log into the rebalance audit record but are
# EXCLUDED from the live composite until their live IC record earns un-gating
# (edit here or set QTS_LIVE_GATED as a comma list; empty string = gate nothing).
DEFAULT_LIVE_GATED = frozenset({
    "event_officer_change", "event_non_reliance", "eightk_freq_anomaly",
    "news_volume_anomaly", "reddit_mention_anomaly",
    "reddit_sentiment", "news_sentiment",
    # Crypto sentiment RISK signals: computed + logged every crypto rebalance, but they
    # touch NOTHING (sizing, overlay, circuit breaker) until each earns un-gating on its
    # forward risk-IC record (qts.crypto.sentiment.research). See the 2026-07-19 design doc.
    "crypto_vol_forecast", "crypto_equity_risk_appetite", "crypto_attention_spike",
})


def live_gated() -> frozenset[str]:
    env = os.environ.get("QTS_LIVE_GATED")
    if env is None:
        return DEFAULT_LIVE_GATED
    return frozenset(x.strip() for x in env.split(",") if x.strip())
