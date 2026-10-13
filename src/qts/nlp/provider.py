"""NLP-factory provider interface (wired seam; not active in v1).

V2 design: the NLP layer is a BATCH JOB submitted to GPU worker via an external job runner, not a
co-located service. Flow:
  1. the main host extracts filing sections from the EDGAR corpus in the PIT store.
  2. the job runner wakes GPU worker (WoL); the job runs SEQUENTIALLY to
     respect the 12 GB VRAM cap:
        a. corpus-wide sentence-embedding diff (Lazy-Prices cosine shift) — bge/e5
        b. FinBERT/DeBERTa sentiment + change-classification
        c. route only the FLAGGED subset to a quantized instruct LLM (qwen3-14b already on
           the box) for hedged-guidance / subtle-tone reasoning
  3. results (PIT-stamped: publication timestamp = filing knowledge_time) rsync back to the
     pool; GPU worker auto-sleeps.
  4. This provider then just READS those precomputed features from the store, as_of-filtered.

Model weights are pinned + hashed per backtest run (leak guard #3): a filing is never
scored before its filing timestamp, and the same frozen weights are used across a run.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..pit.store import PITStore
from ..types import Backtestability, Direction, FeatureSpec
from ..features.base import FeatureProvider, register


class FilingSentiment(FeatureProvider):
    """Reads precomputed 10-K sentiment (qwen3-14b on the GPU worker) from the nlp_features table,
    as_of-filtered (filing_date <= as_of). NaN where no filing known yet -> self-excludes."""
    spec = FeatureSpec(
        name="filing_sentiment",
        direction=Direction.HIGHER_BETTER,
        backtestability=Backtestability.BACKTESTABLE,  # frozen weights + filing-dated => PIT
        description="LLM sentiment over latest 10-K risk-factor/MD&A section (GPU worker Ollama job).",
    )

    def compute(self, store: PITStore, symbols: list[str], as_of: date) -> pd.Series:
        return store.nlp_features_as_of(symbols, "filing_sentiment", as_of)


class LazyPricesDiff(FeatureProvider):
    """Deterministic language-shift magnitude between consecutive 10-K filings (cosine
    distance of nomic-embed-text embeddings). Larger shift historically bearish."""
    spec = FeatureSpec(
        name="lazy_prices_diff",
        direction=Direction.LOWER_BETTER,
        backtestability=Backtestability.BACKTESTABLE,
        description="Cosine distance between consecutive 10-K sections (GPU worker embeddings).",
    )

    def compute(self, store: PITStore, symbols: list[str], as_of: date) -> pd.Series:
        return store.nlp_features_as_of(symbols, "lazy_prices_diff", as_of)


register(FilingSentiment())
register(LazyPricesDiff())
