"""Hard risk limits: per-position caps, name caps, and a drawdown kill-switch.

Pure / testable. No network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_limits(
    weights: dict[str, float],
    max_position: float = 0.10,
    max_names: int | None = None,
) -> dict[str, float]:
    """Cap per-position weight and (optionally) the number of names, then renormalize.

    Procedure:
      1. drop non-positive weights,
      2. if ``max_names`` is set, keep the top-``max_names`` by weight,
      3. iteratively cap at ``max_position`` and redistribute the excess to uncapped
         names pro-rata, preserving the original gross (sum) of the book,
      4. names that cannot absorb redistribution (all at cap) leave the residual as cash.

    Gross is preserved up to the point where every name is at the cap; beyond that the
    book is necessarily under-invested (cash remainder). Returns a dict.
    """
    w = pd.Series({k: float(v) for k, v in weights.items() if v is not None})
    w = w[w > 0]
    if w.empty:
        return {}

    if max_names is not None and len(w) > max_names:
        w = w.sort_values(ascending=False).head(max_names)

    gross = float(w.sum())
    # Infeasible to keep full gross if cap*n < gross -> book caps out, rest is cash.
    target = min(gross, max_position * len(w))

    w = w / w.sum() * target  # start proportional at the achievable gross
    for _ in range(100):
        over = w > max_position + 1e-12
        if not over.any():
            break
        excess = float((w[over] - max_position).sum())
        w[over] = max_position
        under = ~over & (w < max_position - 1e-12)
        if not under.any():
            # everyone left is at the cap; remaining excess becomes cash
            break
        room = (max_position - w[under])
        w[under] = w[under] + excess * (room / room.sum())
    return {k: float(v) for k, v in w.items() if v > 1e-12}


class KillSwitch:
    """Tracks running drawdown off an equity curve; trips at a max drawdown.

    Once tripped it STAYS tripped (latching) — recovery is a deliberate manual reset,
    not something a noisy equity tick should silently undo.
    """

    def __init__(self, max_drawdown: float = 0.25):
        if not 0 < max_drawdown < 1:
            raise ValueError("max_drawdown must be in (0, 1)")
        self.max_drawdown = float(max_drawdown)
        self.tripped = False
        self.peak = float("-inf")
        self.last_drawdown = 0.0

    def check(self, equity_curve) -> bool:
        """Return True if the kill-switch is (now or already) tripped.

        Accepts a scalar latest-equity, a list/array, or a pandas Series. Computes the
        worst peak-to-trough drawdown seen and latches ``tripped`` if it breaches the
        threshold.
        """
        vals = self._coerce(equity_curve)
        if vals.size == 0:
            return self.tripped
        running_peak = np.maximum.accumulate(vals)
        # guard against non-positive equity
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(running_peak > 0, 1.0 - vals / running_peak, 0.0)
        self.peak = float(running_peak[-1])
        self.last_drawdown = float(dd[-1])
        if float(np.nanmax(dd)) >= self.max_drawdown:
            self.tripped = True
        return self.tripped

    def reset(self) -> None:
        self.tripped = False
        self.peak = float("-inf")
        self.last_drawdown = 0.0

    @staticmethod
    def _coerce(x) -> np.ndarray:
        if isinstance(x, pd.Series):
            x = x.to_numpy()
        arr = np.atleast_1d(np.asarray(x, dtype=float))
        return arr[np.isfinite(arr)]
