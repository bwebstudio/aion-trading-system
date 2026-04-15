"""
aion.regime.base
─────────────────
Abstract base class for regime detectors.

STUB — not yet implemented.

The regime detector consumes a MarketSnapshot and returns a RegimeLabel
with a confidence score.  The V1 implementation will be rule-based
(ADX, ATR percentile, Bollinger Width).  A learned classifier will be
added later via the research pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from aion.core.enums import RegimeLabel
from aion.core.models import MarketSnapshot


@dataclass(frozen=True)
class RegimeResult:
    """Output of a regime detector."""

    label: RegimeLabel
    confidence: float  # 0.0 – 1.0
    model_version: str


class RegimeDetector(ABC):
    """Base class for regime detectors."""

    @property
    @abstractmethod
    def detector_id(self) -> str:
        """Unique identifier for this detector implementation."""

    @abstractmethod
    def detect(self, snapshot: MarketSnapshot) -> RegimeResult:
        """Classify the current market regime from the snapshot."""
