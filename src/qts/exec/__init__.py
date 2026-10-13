"""Execution layer — Alpaca paper -> live (seam for v2).

Live Alpaca trading is DEFERRED: `alpaca` is lazy-imported, so this package
imports fine with no keys and without alpaca-py installed. To go live on paper:
    uv pip install alpaca-py
    export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...
"""
from qts.exec.broker import AlpacaBroker, Broker, DryRunBroker, TargetPortfolio
from qts.exec.reconcile import compute_orders

__all__ = [
    "Broker",
    "DryRunBroker",
    "AlpacaBroker",
    "TargetPortfolio",
    "compute_orders",
]
