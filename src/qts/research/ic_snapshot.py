"""Weekly IC-history snapshot — maintains outputs/ic_history.csv, the persistent
per-provider IC record the live rebalance reads for adaptive weights (and the
evidence base for un-gating decisions). Recomputes the trailing RECOMPUTE_DAYS
each run and MERGES over the existing file (older rows are kept, overlapping
rows are refreshed), so history extends indefinitely while each run stays cheap.
Runs from deploy/daily-pipeline.sh on --with-sec nights (Fri/Sun), under the
main-DB lock the pipeline already holds. ALL providers are included — gated ones
too: their IC record accrues here even while they trade at zero weight."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ..config import OUTPUTS_DIR, ensure_dirs
from ..features.base import default_providers
from ..pit.store import PITStore
from .ic import provider_ic

IC_HISTORY = OUTPUTS_DIR / "ic_history.csv"
RECOMPUTE_DAYS = 550   # ~78 weekly rows recomputed per run


def load_ic_history(path=None) -> pd.DataFrame:
    p = path or IC_HISTORY
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df.sort_index()


def update_ic_history(store: PITStore, today: date | None = None,
                      path=None) -> pd.DataFrame:
    ensure_dirs()
    today = today or date.today()
    start = today - timedelta(days=RECOMPUTE_DAYS)
    fresh = provider_ic(store, default_providers(), start, today)
    old = load_ic_history(path)
    # combine_first prefers `fresh` on overlaps and keeps old-only rows/columns
    merged = fresh if old.empty else fresh.combine_first(old)
    merged = merged.sort_index()
    merged.to_csv(path or IC_HISTORY)
    print(f"ic_snapshot: {len(fresh)} fresh rows, {len(merged)} total, "
          f"{merged.shape[1]} providers")
    return merged


def main() -> None:
    with PITStore(read_only=True) as store:
        update_ic_history(store)


if __name__ == "__main__":
    main()
