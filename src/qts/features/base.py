"""Feature-provider contract + registry.

A provider, given (store, symbols, as_of), returns a PIT-correct pd.Series indexed by
symbol. It advertises a FeatureSpec (name, direction, backtestability). The ranker is the
only thing that standardizes/sign-adjusts; providers emit RAW values.

Leak guard #6: providers declare backtestability; the backtest engine refuses
forward_only providers in backtest mode.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from ..pit.store import PITStore
from ..types import Backtestability, Direction, FeatureSpec


class FeatureProvider(ABC):
    spec: FeatureSpec

    @abstractmethod
    def compute(self, store: PITStore, symbols: list[str], as_of: date) -> pd.Series:
        """Return Series(index=symbol, value=raw feature). NaN where unavailable."""
        ...

    @property
    def name(self) -> str:
        return self.spec.name


REGISTRY: dict[str, FeatureProvider] = {}


def register(p: FeatureProvider) -> FeatureProvider:
    REGISTRY[p.spec.name] = p
    return p


def default_providers() -> list[FeatureProvider]:
    """Instantiate + register the v1 provider set (import side-effect populates REGISTRY)."""
    from . import altsignals, events, factors, fundamentals_factors, sentiment, tier1  # noqa: F401
    from ..nlp import provider as _nlp  # noqa: F401  (registers filing_sentiment, lazy_prices_diff)
    return list(REGISTRY.values())


def backtestable_providers() -> list[FeatureProvider]:
    return [p for p in default_providers()
            if p.spec.backtestability == Backtestability.BACKTESTABLE]
