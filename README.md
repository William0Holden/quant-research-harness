# quant-research-harness

A point-in-time, survivorship-aware, **leak-guarded** systematic equity research harness for the
S&P 500. It fuses factor signals, SEC-filing NLP and social/news sentiment into one
cross-sectional ranker, validates it with walk-forward backtests net of realistic costs, and runs
a weekly rebalance against an **Alpaca paper account**. Built June–September 2026.

> **Honest verdict:** this is a research harness, not a proven edge. It is infrastructure for
> testing ideas without fooling yourself (point-in-time data, leak guards, mandatory costs,
> forward IC tracking) — not a money-maker. Paper trading only.
>
> **Not financial advice.** Nothing here is a recommendation to buy or sell anything.

## What it does

- **Bitemporal point-in-time store (DuckDB).** Every fact carries a `knowledge_time`; the only
  read accessor takes `as_of` and filters `knowledge_time <= as_of`, so future leakage is not
  expressible. In the author's deployment it held ~4.3M price bars, ~471k fundamental facts and
  ~409k filing events.
- **Universe:** ~503 S&P 500 constituents (Wikipedia fetch with a vendored fallback CSV) plus
  delisted names. Data sources are key-free: SEC EDGAR, Yahoo prices, FRED.
- **11 live signal providers:** momentum 12-1, low-vol, earnings-yield value, gross-profitability
  quality, net share issuance, insider (Form 4) activity, 8-K intensity, restatement (8-K 4.02)
  flag, short interest, and two NLP signals — 10-K filing sentiment and a "Lazy Prices"
  language-shift (embedding cosine diff). Five more alt-data providers (officer-change and
  non-reliance 8-Ks, 8-K frequency anomaly, news-volume anomaly, Reddit-mention anomaly) are
  computed and audit-logged every run but **live-gated** until they earn an IC track record.
- **LLM sentiment layer:** RSS/news and Reddit collectors write to a separate corpus DB; a remote
  GPU worker scores texts with a local Ollama model over SSH; a nightly aggregation distills
  per-symbol daily signals back into the PIT store.
- **Signal research:** per-provider Spearman IC on exactly the next-bar window the backtest
  trades; trailing, past-only IC weights as an alternative to equal weighting.
- **Walk-forward backtest** with mandatory non-zero spread/slippage/commission and short-borrow
  modeling; refuses forward-only providers.
- **Risk/sizing:** vol targeting + quarter-Kelly (sized off half the estimated edge), position
  caps, FRED macro-regime gross scaling (HY OAS / term spread / VIX) that fails *closed*, and a
  latching kill switch that flattens at 25% peak-to-trough drawdown of recorded broker equity.
- **Execution:** `qts.exec.rebalance` ranks, sizes and hands targets to a `Broker` —
  `AlpacaBroker` (paper) when keys are present, otherwise `DryRunBroker`. Every run writes an
  audited JSON record (weights, orders, equity provenance, gated signals). Orders submit Friday
  after close and fill at Monday open, matching the backtest's next-bar discipline.
- **Crypto trend book** (BTC/ETH, trend-timed) on a *separate* paper account; a guard refuses to
  run if it would share the equity account.
- **Streamlit dashboard:** universe, rankings, NLP signals, portfolio/sizing, regime, backtest,
  Signal IC, and live leak-guard self-checks. Optional password gate (`QTS_DASH_PASSWORD`).

## Architecture

```
 SEC EDGAR / Yahoo / FRED ──► ingest/ ──► pit/ (qts.duckdb, bitemporal, as_of-only reads)
 RSS news / Reddit ─────────► altdata/ ─► corpus.duckdb ──► GPU worker (Ollama, SSH batch)
                                              │                      │
                                              └── altdata.aggregate ◄┘ (PIT-stamped)
 pit/ ──► features/ (providers + ranker) ──► research/ (IC)    backtest/ (walk-forward)
                         │
                         └──► risk/ (vol target, ¼-Kelly, regime, kill switch)
                                  └──► exec/ (Alpaca paper | dry run) ──► outputs/rebalances/*.json
 dashboard/ (Streamlit, read-only DB handle)
```

```
src/qts/
  pit/        bitemporal DuckDB store, survivorship-free universe, S&P 500 constituents
  ingest/     Yahoo prices + SEC fundamentals/events/filings (key-free, resumable)
  altdata/    news + Reddit collectors, ticker linker, corpus -> alt_signals aggregation
  features/   provider contract, factor/event/sentiment providers, composite ranker
  nlp/        filing NLP + sentiment factories (remote GPU batch jobs)
  research/   information coefficient, trailing IC weights, IC snapshots
  backtest/   walk-forward engine, cost model, metrics
  risk/       sizing, regime, limits, kill switch
  exec/       Broker ABC, DryRunBroker, AlpacaBroker (paper), rebalance runner
  crypto/     crypto trend book + crypto sentiment (risk-only, gated)
  dashboard/  Streamlit app
  checks.py   leak-guard self-checks
deploy/       cron wrapper scripts + crontab.example
tests/        pytest suite (hermetic: no data or network required)
```

### Leak guards (enforced structurally)

1. Survivorship-free universe table — no "current tickers" list to misuse.
2. `knowledge_time` on every fact; the accessor requires `as_of`.
3. NLP model weights pinned per run; a filing is never scored before its filing timestamp.
4. Walk-forward only; eligibility gated on real price history.
5. Costs and borrow non-zero by default; zeroing is a conscious override.
6. Providers declare backtestability; the engine refuses forward-only providers in backtests.

## Running it (paper trading only)

Requires Python ≥3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras          # core + dev/live/ml extras
cp .env.example .env          # fill in what you need; everything is optional for research
uv run pytest -q              # hermetic test suite

# 1) Build the PIT store (first run is a full-history pull, roughly an hour)
uv run python -m qts.ingest.run                 # later: --incremental [--no-sec]

# 2) Dashboard
uv run streamlit run src/qts/dashboard/app.py

# 3) Rebalance: dry run by default; --submit sends to Alpaca PAPER iff keys are set
uv run python -m qts.exec.rebalance
uv run python -m qts.exec.rebalance --submit

# Optional: alt-data + NLP
uv run python -m qts.altdata.news
uv run python -m qts.altdata.reddit             # no-op without REDDIT_* creds
uv run python -m qts.altdata.aggregate
uv run python -m qts.ingest.filings
uv run python -m qts.nlp.factory                # needs a GPU worker (QTS_GPU_*)
```

Leak-guard self-check against a built store:

```bash
uv run python -c "from qts.pit.store import PITStore; from qts.checks import run_leak_checks; print(run_leak_checks(PITStore(read_only=True)))"
```

**Scheduling.** `deploy/*.sh` are cron wrappers that load `.env`, respect DuckDB's single-writer
rule via `flock` (`outputs/.qts-db.lock` for the main store, `outputs/.qts-corpus.lock` for the
corpus), and log to `outputs/`. See `deploy/crontab.example` for the reference cadence (UTC):
nightly incremental prices, Friday SEC refresh then the weekly rebalance, Sunday filings + NLP,
hourly news and twice-hourly Reddit collection.

**Configuration.** All credentials and host settings are environment variables — see
`.env.example` for the names. Key toggles:

| Env var | Effect |
|---|---|
| `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` | Alpaca **paper** keys; absent → `DryRunBroker` |
| `APCA_CRYPTO_KEY_ID` / `APCA_CRYPTO_SECRET_KEY` | Separate paper account for the crypto book |
| `QTS_DAILY_REBALANCE` | `1` enables the (default-off) Mon–Thu daily rebalance |
| `QTS_LIVE_GATED` | Comma-separated providers excluded from the live composite (overrides the default 5) |
| `QTS_GPU_HOST` etc. | SSH target for the remote Ollama scorer |

Gated signals are un-gated only after a positive, statistically meaningful trailing IC on the
dashboard's Signal IC tab. Rolling-window anomaly signals need ~21 days of corpus history before
they produce values (they compute as NaN until then, by design).

## Known limitations

- Point-in-time index *membership* is not reconstructed: current constituents are treated as
  members back to 2000, so index entrants leak retroactively into historical backtests.
- Delisted-name price history is incomplete.
- Sentiment depends on a separate GPU box being reachable; scoring skips cleanly when it isn't.
- No result here should be read as evidence of alpha; see the verdict at the top.

