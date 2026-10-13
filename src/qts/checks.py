"""Runtime leak-guard self-checks. Surfaced live in the dashboard so the discipline
layer is verifiable, not just asserted in docs."""
from __future__ import annotations

from datetime import date

import pandas as pd

from .config import CostModel
from .pit.store import PITStore
from .types import Backtestability, Direction, FeatureSpec


def run_leak_checks(store: PITStore) -> pd.DataFrame:
    out = []

    def add(name, passed, detail):
        out.append({"check": name, "passed": bool(passed), "detail": detail})

    # 1. as-of refuses future price bars
    p = store.prices_as_of(["AAPL"], date(2020, 1, 1), lookback_days=60)
    mx = pd.to_datetime(p["dt"]).max() if len(p) else None
    add("prices: no future bars (as_of 2020-01-01)",
        mx is not None and mx <= pd.Timestamp("2020-01-01"),
        f"max dt = {mx.date() if mx is not None else 'none'}")

    # 2. fundamentals are filing-dated (no period-end leak)
    f = store.fundamentals_as_of(["AAPL"], ["NetIncomeLoss"], date(2015, 1, 1), annual_only=True)
    fd = pd.to_datetime(f["filing_date"]).max() if len(f) else None
    add("fundamentals: filing_date <= as_of (2015-01-01)",
        fd is not None and fd <= pd.Timestamp("2015-01-01"),
        f"latest known filing = {fd.date() if fd is not None else 'none'}")

    # 3. survivorship-free universe
    alive_22 = set(store.universe_as_of(date(2022, 6, 1)).symbol)
    alive_24 = set(store.universe_as_of(date(2024, 1, 1)).symbol)
    add("universe: delisted name present before death, absent after",
        ("FRC" in alive_22) and ("FRC" not in alive_24),
        f"FRC in 2022={'FRC' in alive_22}, in 2024={'FRC' in alive_24}")

    # 4. backtest refuses forward-only providers
    class _Fwd:
        spec = FeatureSpec("dummy_forward", Direction.HIGHER_BETTER,
                           Backtestability.FORWARD_ONLY, "")
        def compute(self, *a, **k):
            return pd.Series(dtype=float)
    refused = False
    try:
        from .backtest.engine import run_backtest
        run_backtest(store, [_Fwd()], date(2020, 1, 1), date(2021, 1, 1))
    except ValueError as e:
        refused = "forward_only" in str(e)
    except Exception:
        refused = False
    add("backtest: forward_only providers refused", refused,
        "engine raises before running")

    # 5. costs non-zero by default
    cm = CostModel()
    add("costs: per-side friction non-zero by default", cm.per_side_bps() > 0,
        f"{cm.per_side_bps():.1f} bps/side (spread+slippage+commission)")

    return pd.DataFrame(out)
