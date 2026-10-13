"""Macro regime conditioner from FRED (key-free).

Uses the public fredgraph CSV endpoint, which needs no API key:

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>

Series used:
    BAMLH0A0HYM2  ICE BofA US High Yield OAS (credit-spread stress gauge)
    T10Y2Y        10y minus 2y Treasury term spread (curve / recession gauge)
    VIXCLS        CBOE VIX (equity implied vol)

PIT note: FRED publishes daily macro series with ~same/next-day latency. The CSV carries
only the *observation date*, not a release timestamp, so we use observation_date <= as_of
as the knowledge filter. This is a mild forward-looking optimism (today's value may not be
released until tomorrow), acceptable for a weekly-rebalanced regime overlay; for a strict
backtest, lag ``as_of`` by one business day before calling.

FAIL-CLOSED POLICY (2026-07-28). This brake used to return gross_multiplier=1.0 whenever a
series could not be read -- the 2026-07-24 rebalance sized the book at full gross with all
three series NaN. An unreadable series does not mean "risk-on", it means "unknown", and
unknown is not a licence to run full exposure. Two layers now stand between a FRED hiccup
and a mis-sized book:

  1. Last-known-good disk cache. Every successful fetch persists the trailing history to
     outputs/regime_cache.json; a failed fetch falls back to it. A transient outage
     therefore changes NOTHING about the book -- the right outcome, since flattening on
     every hiccup would buy whipsaw turnover against an overlay consulted once a week.
  2. Staleness => degraded. If the newest observation visible at as_of is older than
     MAX_STALENESS_DAYS, we genuinely do not know the current regime, and gross is capped
     at DEGRADED_GROSS_MULTIPLIER.

Only hy_oas and vix feed tripwires, so only their absence degrades; term_spread is
reported for context and its loss must not de-risk the book.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from ..config import OUTPUTS_DIR

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

SERIES = {
    "hy_oas": "BAMLH0A0HYM2",
    "term_spread": "T10Y2Y",
    "vix": "VIXCLS",
}

# Series that actually feed a tripwire. term_spread is reported but unused in the
# decision, so losing it must not degrade the book.
DECISION_SERIES = ("hy_oas", "vix")

# --- thresholds (documented) ------------------------------------------------
# Risk-off if EITHER credit stress (HY OAS in the top quartile of its trailing 2y
# distribution) OR an outright equity-vol spike (VIX > 25). Top-quartile credit
# spread is a relative, self-calibrating stress signal; VIX>25 is an absolute floor
# that catches fast equity shocks the slower-moving credit market hasn't repriced yet.
VIX_RISK_OFF = 25.0
HY_OAS_QUANTILE = 0.75
TRAILING_WINDOW_DAYS = 730  # ~2 years

# An observation older than this means we do not know the current regime. FRED's daily
# series gap by at most ~4 calendar days over a holiday weekend, so 10 days does not
# trip on quiet data -- it trips when a series is genuinely broken. Applied uniformly
# to live and cached reads: one rule, one test.
MAX_STALENESS_DAYS = 10

# Gross cap while the regime is unknown. Deliberately equal to the one-tripwire
# multiplier: an unknown tripwire is worth exactly as much de-risking as a fired one.
# Bounded, never to cash -- a data outage should not liquidate the book.
DEGRADED_GROSS_MULTIPLIER = 0.6

CACHE_KEEP_DAYS = 800  # a little over the 2y trailing window the HY quantile needs
_HEADERS = {"User-Agent": "qts-research regime-overlay"}


def cache_path() -> Path:
    return Path(os.environ.get("QTS_REGIME_CACHE", OUTPUTS_DIR / "regime_cache.json"))


def fetch_fred_series(series_id: str, as_of: date | datetime, timeout: float = 20.0) -> pd.Series:
    """Fetch one FRED series as a date-indexed float Series, filtered to <= as_of.

    Missing observations (FRED encodes them as '.') are dropped.
    """
    url = FRED_CSV.format(sid=series_id)
    r = requests.get(url, headers=_HEADERS, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # First column is the date (named observation_date on the modern endpoint, DATE on
    # the legacy one); second column is the value (named after the series id).
    date_col, val_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")  # '.' -> NaN
    s = df.dropna(subset=[date_col]).set_index(date_col)[val_col].dropna()
    s.index = s.index.normalize()
    cutoff = pd.Timestamp(as_of).normalize()
    s = s[s.index <= cutoff]
    s.name = series_id
    return s.sort_index()


def _empty_series() -> pd.Series:
    # DatetimeIndex, not the default RangeIndex: the trailing-window comparison in
    # regime_state is index >= Timestamp, and a RangeIndex made that raise -- which
    # nuked the WHOLE regime state on one bad fetch (2026-07-03).
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def _latest(s: pd.Series) -> float:
    return float(s.iloc[-1]) if len(s) else float("nan")


# --- last-known-good cache --------------------------------------------------

def _load_cache() -> dict:
    p = cache_path()
    try:
        if p.exists():
            blob = json.loads(p.read_text())
            return blob.get("series", {}) if isinstance(blob, dict) else {}
    except Exception:  # noqa: BLE001  a corrupt cache is a cache miss, never a crash
        pass
    return {}


def _write_cache(series: dict) -> None:
    """Atomic replace: a half-written cache during a crash would poison every later read."""
    p = cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".regime_cache-", suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump({"series": series}, fh)
        os.replace(tmp, p)
    except Exception:  # noqa: BLE001  cache is an optimisation; never fail the brake on it
        pass


def cache_series(key: str, sid: str, s: pd.Series) -> None:
    """Persist the trailing CACHE_KEEP_DAYS of a freshly fetched series."""
    if s is None or s.empty:
        return
    keep = s[s.index >= (s.index.max() - pd.Timedelta(days=CACHE_KEEP_DAYS))]
    blob = _load_cache()
    blob[key] = {
        "sid": sid,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "obs": {d.strftime("%Y-%m-%d"): float(v) for d, v in keep.items()},
    }
    _write_cache(blob)


def cached_series(key: str) -> pd.Series:
    """Last-known-good series from disk, or an empty Series when absent/unreadable."""
    entry = _load_cache().get(key)
    if not entry or not entry.get("obs"):
        return _empty_series()
    try:
        obs = entry["obs"]
        s = pd.Series(
            [float(v) for v in obs.values()],
            index=pd.DatetimeIndex([pd.Timestamp(d) for d in obs.keys()]),
        )
        return s.sort_index()
    except Exception:  # noqa: BLE001
        return _empty_series()


def _fetch_with_retry(sid: str, as_of, retries: int = 1, sleep_s: float = 2.0) -> pd.Series:
    """One retry absorbs the transient FRED hiccups that a weekly risk brake
    should not fail on (the 2026-07-03 rebalance lost its regime state to one)."""
    import time as _time
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fetch_fred_series(sid, as_of)
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries:
                _time.sleep(sleep_s)
    raise last


def _resolve_series(key: str, sid: str, as_of) -> tuple[pd.Series, str, bool]:
    """(series filtered to as_of, source, live_fetch_failed).

    source is "live" | "cache" | "missing". A live fetch that succeeds also refreshes
    the on-disk cache, so the fallback is always as good as the last healthy run.
    """
    cutoff = pd.Timestamp(as_of).normalize()
    try:
        s = _fetch_with_retry(sid, as_of)
        cache_series(key, sid, s)
        return s, "live", False
    except Exception:  # noqa: BLE001  network/parse failure -> fall back to disk
        s = cached_series(key)
        s = s[s.index <= cutoff]
        if s.empty:
            return _empty_series(), "missing", True
        return s, "cache", True


def _is_stale(s: pd.Series, as_of) -> bool:
    """True when the newest observation we can see is too old to describe 'now'."""
    if s.empty:
        return True
    age = (pd.Timestamp(as_of).normalize() - s.index.max()).days
    return age > MAX_STALENESS_DAYS


def prefetch_series(as_of: date | datetime | None = None) -> dict[str, pd.Series]:
    """Fetch every FRED series ONCE, for injection into `regime_state(series=...)`.

    A backtest evaluates the regime at every rebalance date. Without this, a 4-year
    weekly run would make ~750 HTTP calls to FRED for data it could have fetched once:
    each series is a full history, and `regime_state` already filters it to as_of.

    Falls back to the on-disk cache per series, so an offline backtest still runs with
    the last healthy history rather than silently degrading every date.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc).date()
    out: dict[str, pd.Series] = {}
    for key, sid in SERIES.items():
        s, _source, _failed = _resolve_series(key, sid, as_of)
        out[key] = s
    return out


def regime_state(
    as_of: date | datetime | None = None,
    series: dict[str, pd.Series] | None = None,
) -> dict:
    """Compute the macro regime at ``as_of`` (default: today).

    ``series`` (optional): pre-fetched {key: full Series} from `prefetch_series`, reused
    across many as_of dates instead of re-hitting FRED per call. Each is re-filtered to
    <= as_of here, so injecting a full history is still point-in-time correct.

    Returns a dict:
        hy_oas, term_spread, vix     -> latest level known by as_of (float or NaN)
        hy_oas_q75                   -> trailing-2y 75th pct of HY OAS
        risk_off                     -> bool (credit top-quartile OR VIX>25)
        degraded                     -> bool (a DECISION series is unknown or stale)
        series_source                -> {key: "live"|"cache"|"injected"|"missing"}
        stale_series                 -> keys whose newest observation is too old
        fetch_errors                 -> keys whose live fetch failed (cache may have covered it)
        gross_multiplier             -> float in [0.3, 1.0], gross-exposure scaler

    gross_multiplier mapping (documented):
        risk-on                   -> 1.0
        risk-off (one tripwire)   -> 0.6
        risk-off (both tripwires) -> 0.3
        degraded read             -> capped at DEGRADED_GROSS_MULTIPLIER (0.6)
    The multiplier degrades gross exposure rather than going fully flat, so neither a
    single noisy print nor a data outage can whipsaw the book to cash.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc).date()

    levels: dict[str, float] = {}
    resolved: dict[str, pd.Series] = {}
    sources: dict[str, str] = {}
    fetch_errors: list[str] = []
    stale: list[str] = []
    cutoff = pd.Timestamp(as_of).normalize()

    for key, sid in SERIES.items():
        if series is not None:
            # Injected history: re-filter to as_of so a pre-fetched series stays PIT.
            s = series.get(key, _empty_series())
            s = s[s.index <= cutoff]
            source = "injected" if len(s) else "missing"
            if s.empty:
                fetch_errors.append(key)
        else:
            s, source, failed = _resolve_series(key, sid, as_of)
            if failed:
                fetch_errors.append(key)
        if _is_stale(s, as_of):
            stale.append(key)
        resolved[key] = s
        sources[key] = source
        levels[key] = _latest(s)

    hy = resolved["hy_oas"]
    hy_recent = hy[hy.index >= (pd.Timestamp(as_of) - pd.Timedelta(days=TRAILING_WINDOW_DAYS))]
    hy_q = float(hy_recent.quantile(HY_OAS_QUANTILE)) if len(hy_recent) >= 30 else float("nan")

    # A tripwire is evaluable only when its series is fresh AND its level is a number.
    # hy_oas additionally needs enough trailing history to have a quantile at all.
    credit_known = ("hy_oas" not in stale) and pd.notna(levels["hy_oas"]) and pd.notna(hy_q)
    vol_known = ("vix" not in stale) and pd.notna(levels["vix"])

    credit_stress = bool(credit_known and levels["hy_oas"] >= hy_q)
    vol_spike = bool(vol_known and levels["vix"] > VIX_RISK_OFF)
    risk_off = credit_stress or vol_spike

    if not risk_off:
        gross = 1.0
    elif credit_stress and vol_spike:
        gross = 0.3
    else:
        gross = 0.6

    # Fail closed: an unknown decision series caps gross exactly as a fired tripwire does,
    # and a tripwire that fires DURING a degraded read still gets the stricter multiplier.
    degraded = not (credit_known and vol_known)
    if degraded:
        gross = min(gross, DEGRADED_GROSS_MULTIPLIER)

    return {
        "as_of": str(as_of),
        "hy_oas": levels["hy_oas"],
        "term_spread": levels["term_spread"],
        "vix": levels["vix"],
        "hy_oas_q75": hy_q,
        "credit_stress": credit_stress,
        "vol_spike": vol_spike,
        "risk_off": risk_off,
        "degraded": degraded,                # unknown regime -> gross was capped
        "series_source": sources,            # where each series actually came from
        "stale_series": stale,               # newest observation older than the window
        "gross_multiplier": float(gross),
        "fetch_errors": fetch_errors,        # non-empty = degraded read, visible in audit
    }


if __name__ == "__main__":  # live smoke test (network OK here)
    state = regime_state()
    print("Live FRED regime_state:")
    print(json.dumps(state, indent=2))
