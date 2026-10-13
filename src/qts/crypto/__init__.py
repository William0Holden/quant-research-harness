"""QTS-Crypto: trend-timed BTC/ETH with a sentiment RISK overlay.

Deliberately NOT a cross-sectional coin ranker. That design was falsified on the real
tradable universe before any production code was written -- every price signal regressed
against the crypto market factor showed beta ~1.0, R^2 ~0.87, and no significant alpha.

The honest claim attached to this package: RISK-MANAGED CRYPTO BETA. Trend-timed exposure
that targets roughly buy-and-hold returns at roughly half the drawdown. Not alpha.
"""
