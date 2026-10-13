"""Core types shared across modules."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum


class Backtestability(str, Enum):
    """Leak guard #6: providers declare whether they have clean PIT history."""
    BACKTESTABLE = "backtestable"      # free PIT history exists; allowed in backtest
    FORWARD_ONLY = "forward_only"      # collect-forward only; refused in backtest mode


class Direction(str, Enum):
    HIGHER_BETTER = "higher_better"    # raw signal: higher = more bullish
    LOWER_BETTER = "lower_better"      # raw signal: lower = more bullish (flipped in z-score)


@dataclass(frozen=True)
class UniverseMember:
    symbol: str
    cik: str | None
    name: str
    active_from: date
    active_to: date | None  # None = still active
    delisted: bool


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata a provider advertises about the column it emits."""
    name: str
    direction: Direction
    backtestability: Backtestability
    description: str = ""
