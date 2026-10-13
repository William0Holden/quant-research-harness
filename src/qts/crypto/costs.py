"""Crypto cost model + the ledger we keep because the paper account will not.

VERIFIED EMPIRICALLY AGAINST THE LIVE API (2026-07-14):
Alpaca's paper account charges NO crypto fees. A qty-based $12.90 BTC buy showed an implied
fee of -3.3bps against a published 25bps taker rate. Paper also does not model queue
position, does not check order size against real liquidity, and fills a buy limit only when
the entire ask comes DOWN to your bid.

So paper P&L is inflated by the full round-trip cost and cannot validate this strategy.
We therefore compute what we WOULD have paid on the real schedule, every fill, and carry a
cost-adjusted NAV beside the broker's reported equity. Without this the single most
important variable in the project would be measured by a simulator that flatters it.

SPREAD CAPTURE IS NOT REAL -- DO NOT PUT IT IN THIS MODEL.
Passive fills are signed against you: P(fill | price moved against you) ~= 0.99, versus
P(fill | it moved your way) ~= 0.018 (DeLise, arXiv 2407.16527). You "capture" the
half-spread and hand ~90% of it straight back as adverse markout. The honest model is:

    E[passive edge] = (taker_fee - maker_fee)        # +10bps  <- REAL, the only sure term
                    + half_spread - adverse_markout  # ~= 0    <- MODEL AS ZERO
                    - P(no fill) * opportunity_cost  # signed against you

An earlier draft of this design modelled maker cost as `15 - half_spread/2`, which produced
NEGATIVE costs on wide-spread coins -- i.e. it claimed the venue would pay us to trade. That
was the exact error the literature warns produces backtests "on a completely different scale"
from reality. Maker cost here is the FEE, flat. The strategy earns its keep on the 10bps fee
differential and on an alpha horizon (months) orders of magnitude longer than the
adverse-selection horizon (seconds) -- not on spread capture.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Alpaca crypto tier 1 (0-$100k 30d volume), verified from the live account: crypto_tier=1.
MAKER_FEE_BPS = 15.0
TAKER_FEE_BPS = 25.0


@dataclass(frozen=True)
class CryptoCostModel:
    """Cost per side, in bps. Taker crosses the spread; maker pays the fee only."""
    maker_fee_bps: float = MAKER_FEE_BPS
    taker_fee_bps: float = TAKER_FEE_BPS

    def taker_bps(self, half_spread_bps: float) -> float:
        """Market/marketable order: fee + the half-spread you cross."""
        return self.taker_fee_bps + max(0.0, float(half_spread_bps))

    def maker_bps(self, half_spread_bps: float = 0.0) -> float:
        """Resting limit order: the FEE. No spread credit -- adverse selection eats it.

        half_spread_bps is accepted (and ignored) so callers can pass it uniformly; the
        signature documents that we were offered the spread and declined to book it.
        """
        return self.maker_fee_bps


@dataclass
class Fill:
    """One fill, with everything needed to audit execution after the fact."""
    ts: str
    symbol: str
    side: str                    # buy | sell
    qty: float
    fill_price: float
    arrival_mid: float           # mid at the moment we DECIDED -- snapshot it or lose it
    liquidity: str               # maker | taker | unknown
    half_spread_bps: float
    fee_bps_modelled: float      # what the real schedule WOULD charge
    fee_usd_modelled: float
    fee_usd_broker: float | None = None   # what the broker actually charged (paper: 0.0)

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.fill_price

    @property
    def slippage_bps(self) -> float:
        """Signed implementation shortfall vs arrival mid, in bps. Positive = we paid up."""
        if not self.arrival_mid:
            return float("nan")
        sign = 1.0 if self.side == "buy" else -1.0
        return sign * (self.fill_price / self.arrival_mid - 1.0) * 1e4

    def total_cost_bps(self) -> float:
        """Modelled fee + realised slippage vs arrival. The number that is actually true."""
        s = self.slippage_bps
        return self.fee_bps_modelled + (0.0 if s != s else s)


@dataclass
class CostLedger:
    """Accumulates modelled costs so a cost-adjusted NAV can be carried beside the broker's.

    The broker's paper equity does not subtract crypto fees. This does.
    """
    cost_model: CryptoCostModel = field(default_factory=CryptoCostModel)
    fills: list[Fill] = field(default_factory=list)
    # Optional JSONL file. When set, every prior fill is loaded at construction and every
    # new fill is appended, so the modelled-fee total spans the LIFE of the book. Without
    # this the ledger was rebuilt empty each run and `adjusted_equity` only ever subtracted
    # the current run's fees -- the "honest NAV" forgot 99% of the costs it existed to track.
    path: Path | None = None
    # Fills recorded by THIS process (a subset of `fills` when a file was loaded).
    run_fills: list[Fill] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = Path(self.path)
            if self.path.exists():
                for line in self.path.read_text().splitlines():
                    line = line.strip()
                    if line:
                        self.fills.append(Fill(**json.loads(line)))

    def record(self, symbol: str, side: str, qty: float, fill_price: float,
               arrival_mid: float, half_spread_bps: float, liquidity: str = "unknown",
               fee_usd_broker: float | None = None) -> Fill:
        fee_bps = (self.cost_model.maker_bps(half_spread_bps) if liquidity == "maker"
                   else self.cost_model.taker_bps(half_spread_bps))
        notional = abs(qty) * fill_price
        f = Fill(
            ts=datetime.now(timezone.utc).isoformat(), symbol=symbol, side=side, qty=qty,
            fill_price=fill_price, arrival_mid=arrival_mid, liquidity=liquidity,
            half_spread_bps=half_spread_bps, fee_bps_modelled=fee_bps,
            fee_usd_modelled=notional * fee_bps / 1e4, fee_usd_broker=fee_usd_broker,
        )
        self.fills.append(f)
        self.run_fills.append(f)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(asdict(f)) + "\n")
        return f

    def total_fee_usd(self) -> float:
        return sum(f.fee_usd_modelled for f in self.fills)

    def total_notional(self) -> float:
        return sum(f.notional for f in self.fills)

    def adjusted_equity(self, broker_equity: float) -> float:
        """Broker equity minus the fees the broker never charged us. The honest NAV."""
        return float(broker_equity) - self.total_fee_usd()

    def maker_rate(self) -> float:
        """Share of notional filled as maker. Institutional target for a patient schedule
        is ~75%. There is no post_only flag on Alpaca, so this MUST be measured from the
        fee actually charged, never assumed."""
        tot = self.total_notional()
        if tot <= 0:
            return float("nan")
        return sum(f.notional for f in self.fills if f.liquidity == "maker") / tot

    def summary(self) -> dict:
        """Lifetime summary (everything loaded from `path` plus this run)."""
        return self._summarize(self.fills)

    def run_summary(self) -> dict:
        """This process's fills only -- what the current rebalance did."""
        return self._summarize(self.run_fills)

    @staticmethod
    def _summarize(fills: list[Fill]) -> dict:
        n = len(fills)
        tot = sum(f.notional for f in fills)
        fee = sum(f.fee_usd_modelled for f in fills)
        slips = [f.slippage_bps for f in fills if f.slippage_bps == f.slippage_bps]
        return {
            "n_fills": n,
            "notional_usd": round(tot, 2),
            "fee_usd_modelled": round(fee, 4),
            "fee_bps_effective": round(fee / tot * 1e4, 2) if tot else None,
            "maker_rate": (round(sum(f.notional for f in fills if f.liquidity == "maker") / tot, 4)
                           if tot > 0 else None),
            "mean_slippage_bps": round(sum(slips) / len(slips), 2) if slips else None,
        }
