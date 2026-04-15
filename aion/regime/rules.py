"""
aion.regime.rules
──────────────────
Rule-based regime detector (V1).

Classification uses FeatureVector fields already computed by the pipeline.
No ML, no external data sources, no lookahead.

Rules (applied in priority order)
───────────────────────────────────
1. UNKNOWN      — volatility_percentile_20 is None (cannot classify)
2. COMPRESSION  — volatility_percentile_20 < LOW_PERCENTILE (0.20)
3. TREND_UP     — volatility_percentile_20 >= HIGH_PERCENTILE (0.75)
                  and return_5 > +TREND_RETURN_THRESHOLD (0.0002)
4. TREND_DOWN   — volatility_percentile_20 >= HIGH_PERCENTILE (0.75)
                  and return_5 < -TREND_RETURN_THRESHOLD (0.0002)
5. EXPANSION    — volatility_percentile_20 >= HIGH_PERCENTILE (0.75)
                  with no strong directional bias in return_5
6. RANGE        — everything else (LOW_PERCENTILE <= vp < HIGH_PERCENTILE)

Confidence
───────────
A scalar [0.0, 1.0] expressing how clearly the snapshot fits the label:
  UNKNOWN     → 0.0 (always)
  COMPRESSION → 1.0 at vp=0; 0.0 at the LOW_PERCENTILE boundary
  RANGE       → 1.0 at midpoint between thresholds; 0.0 at either boundary
  TREND_UP/D  → proportional to |return_5|, capped at 1.0
  EXPANSION   → proportional to how far above HIGH_PERCENTILE

Limitations (V1)
────────────────
- No lookback trend confirmation (e.g. consecutive higher highs).
- No regime persistence — labels can flip bar to bar.
- CHAOTIC and REVERSAL from the enum are not assigned in this version.
- Treat as a coarse signal filter, not a stand-alone trading signal.
"""

from __future__ import annotations

from aion.core.enums import RegimeLabel
from aion.core.models import MarketSnapshot
from aion.regime.base import RegimeDetector, RegimeResult


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds  (module-level for transparency and testability)
# ─────────────────────────────────────────────────────────────────────────────

LOW_PERCENTILE: float = 0.20
"""volatility_percentile_20 below this → COMPRESSION."""

HIGH_PERCENTILE: float = 0.75
"""volatility_percentile_20 at or above this → directional / expansion region."""

TREND_RETURN_THRESHOLD: float = 0.0002
"""Minimum |return_5| to classify as trending rather than noisy expansion."""

_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# RuleBasedRegimeDetector
# ─────────────────────────────────────────────────────────────────────────────


class RuleBasedRegimeDetector(RegimeDetector):
    """
    Classifies market regime using hard-coded thresholds on FeatureVector.

    Instantiate once; call detect() on each MarketSnapshot.

    Parameters
    ----------
    low_percentile:
        volatility_percentile_20 below this → COMPRESSION.  Default 0.20.
    high_percentile:
        volatility_percentile_20 at or above this → directional region.
        Default 0.75.
    trend_return_threshold:
        Minimum |return_5| to classify as TREND_UP/DOWN vs EXPANSION.
        Default 0.0002 (~2 pips for EURUSD).
    """

    def __init__(
        self,
        low_percentile: float = LOW_PERCENTILE,
        high_percentile: float = HIGH_PERCENTILE,
        trend_return_threshold: float = TREND_RETURN_THRESHOLD,
    ) -> None:
        self._low = low_percentile
        self._high = high_percentile
        self._trend_ret = trend_return_threshold

    @property
    def detector_id(self) -> str:
        return "rule_based_v1"

    def detect(self, snapshot: MarketSnapshot) -> RegimeResult:
        fv = snapshot.feature_vector
        vp = fv.volatility_percentile_20
        r5 = fv.return_5

        # ── UNKNOWN: required features missing ───────────────────────────────
        if vp is None:
            return RegimeResult(
                label=RegimeLabel.UNKNOWN,
                confidence=0.0,
                model_version=_VERSION,
            )

        # ── COMPRESSION: very low volatility ─────────────────────────────────
        if vp < self._low:
            # 1.0 furthest from boundary (vp→0); 0.0 at boundary (vp→low)
            confidence = round(max(0.0, 1.0 - (vp / self._low)), 4)
            return RegimeResult(
                label=RegimeLabel.COMPRESSION,
                confidence=confidence,
                model_version=_VERSION,
            )

        # ── High-volatility region (vp >= HIGH_PERCENTILE) ───────────────────
        if vp >= self._high:
            span = 1.0 - self._high
            vol_confidence = round(
                min(1.0, (vp - self._high) / (span + 1e-9)), 4
            )

            if r5 is None:
                return RegimeResult(
                    label=RegimeLabel.EXPANSION,
                    confidence=vol_confidence,
                    model_version=_VERSION,
                )

            if r5 > self._trend_ret:
                trend_conf = round(
                    min(1.0, abs(r5) / (self._trend_ret * 5)), 4
                )
                return RegimeResult(
                    label=RegimeLabel.TREND_UP,
                    confidence=trend_conf,
                    model_version=_VERSION,
                )

            if r5 < -self._trend_ret:
                trend_conf = round(
                    min(1.0, abs(r5) / (self._trend_ret * 5)), 4
                )
                return RegimeResult(
                    label=RegimeLabel.TREND_DOWN,
                    confidence=trend_conf,
                    model_version=_VERSION,
                )

            # High vol but |return_5| <= threshold → EXPANSION
            return RegimeResult(
                label=RegimeLabel.EXPANSION,
                confidence=vol_confidence,
                model_version=_VERSION,
            )

        # ── RANGE: mid volatility (LOW_PERCENTILE <= vp < HIGH_PERCENTILE) ───
        mid = (self._low + self._high) / 2.0
        half_span = mid - self._low  # = (high - low) / 2
        dist_from_mid = abs(vp - mid) / (half_span + 1e-9)
        confidence = round(max(0.0, min(1.0, 1.0 - dist_from_mid)), 4)
        return RegimeResult(
            label=RegimeLabel.RANGE,
            confidence=confidence,
            model_version=_VERSION,
        )
