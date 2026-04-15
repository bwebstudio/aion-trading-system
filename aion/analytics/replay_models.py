"""
aion/analytics/replay_models.py
─────────────────────────────────
Frozen Pydantic models for replay analytics.

Models:
  ReplayMetrics           — aggregate numeric statistics from a replay run
  BreakdownRow            — one row in a grouped breakdown table
  ReplayReport            — full report with per-group breakdowns
  SweepPoint              — one parameter combination for a sweep
  SweepResult             — metrics from one sweep run
  SweepComparison         — all sweep results with ranking helpers
"""

from __future__ import annotations

from pydantic import BaseModel

from aion.core.enums import TradeDirection


class ReplayMetrics(BaseModel, frozen=True):
    """
    Aggregate statistics computed from a replay run.

    Aggregation rules:
      - activation_rate       = entry_activated_count / candidate_count
                                (None when candidate_count == 0)
      - win_rate_on_activated = win_count / entry_activated_count
                                (None when entry_activated_count == 0)
                                Denominator is activated entries only;
                                ENTRY_NOT_ACTIVATED does not count.
      - avg_mfe / avg_mae     = mean over labels where entry_activated=True
                                (None when no activations)
      - avg_bars_to_entry     = mean bars_to_entry over activated labels
      - avg_bars_to_resolution= mean bars_to_resolution over activated labels
    """

    total_records: int
    no_trade_count: int
    candidate_count: int
    insufficient_data_count: int

    # label counts
    total_labeled: int
    entry_activated_count: int
    win_count: int
    loss_count: int
    timeout_count: int
    entry_not_activated_count: int

    # rates (None when denominator is zero)
    activation_rate: float | None
    win_rate_on_activated: float | None

    # averages over activated entries (None when no activations)
    avg_mfe: float | None
    avg_mae: float | None
    avg_bars_to_entry: float | None
    avg_bars_to_resolution: float | None


class BreakdownRow(BaseModel, frozen=True):
    """One row of a grouped breakdown table.

    For session breakdowns: no_trade_count is always 0 because NO_TRADE
    records carry no session information in the replay result.

    For reason_code breakdowns: candidate_count and win/loss counts are
    always 0 because only NO_TRADE records have a reason_code.

    For direction breakdowns: all records are labels (candidates);
    no_trade_count is always 0.
    """

    group_key: str
    record_count: int
    candidate_count: int
    no_trade_count: int
    entry_activated_count: int
    win_count: int
    loss_count: int
    timeout_count: int
    not_activated_count: int
    win_rate: float | None          # win_count / entry_activated_count
    activation_rate: float | None   # entry_activated_count / candidate_count


class ReplayReport(BaseModel, frozen=True):
    """Full structured analytics report from a replay run.

    Breakdowns:
      by_session    — CANDIDATE records grouped by session_name
      by_regime     — all records grouped by regime_label
      by_reason_code— NO_TRADE records grouped by reason_code (desc by count)
      by_direction  — labels grouped by trade direction

    top_reason_codes — [(reason_code, count), ...] sorted descending by count.
    """

    total_records: int
    overall_metrics: ReplayMetrics
    by_session: list[BreakdownRow]
    by_regime: list[BreakdownRow]
    by_reason_code: list[BreakdownRow]
    by_direction: list[BreakdownRow]
    top_reason_codes: list[tuple[str, int]]


class SweepPoint(BaseModel, frozen=True):
    """One parameter combination for a replay parameter sweep.

    Fields that are None imply the corresponding filter is not applied:
      allowed_sessions=None              → no SessionFilter
      max_spread_pips=None               → no SpreadFilter
      direction_bias=None                → both directions allowed
      max_retest_penetration_points=None → no retest depth check
    """

    label: str
    min_range_pips: float = 5.0
    max_range_pips: float = 40.0
    direction_bias: TradeDirection | None = None
    require_completed_range: bool = True
    allowed_sessions: frozenset[str] | None = None
    max_spread_pips: float | None = None
    max_retest_penetration_points: float | None = None
    stop_pips: float = 10.0
    target_pips: float = 20.0
    max_label_bars: int = 50


class SweepResult(BaseModel, frozen=True):
    """Metrics from one parameter combination in a sweep."""

    sweep_point: SweepPoint
    metrics: ReplayMetrics


class SweepComparison(BaseModel, frozen=True):
    """All results from a parameter sweep, with ranking helpers.

    Methods return new sorted lists; the model itself is immutable.
    """

    results: list[SweepResult]

    def ranked_by_win_rate(self) -> list[SweepResult]:
        """Sort by win_rate_on_activated descending; None ranks last."""

        def _key(r: SweepResult) -> float:
            v = r.metrics.win_rate_on_activated
            return v if v is not None else -1.0

        return sorted(self.results, key=_key, reverse=True)

    def ranked_by_candidate_count(self) -> list[SweepResult]:
        """Sort by candidate_count descending."""
        return sorted(self.results, key=lambda r: r.metrics.candidate_count, reverse=True)

    def ranked_by_activation_rate(self) -> list[SweepResult]:
        """Sort by activation_rate descending; None ranks last."""

        def _key(r: SweepResult) -> float:
            v = r.metrics.activation_rate
            return v if v is not None else -1.0

        return sorted(self.results, key=_key, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# VWAP Fade sweep models
# ─────────────────────────────────────────────────────────────────────────────


class VWAPSweepPoint(BaseModel, frozen=True):
    """One parameter combination for a VWAP Fade parameter sweep.

    Fields that are None imply the corresponding filter is not applied:
      direction_bias=None    → both directions allowed
      max_spread_pips=None   → no SpreadFilter

    session_name mirrors VWAPFadeDefinition.session_name:
      'LONDON', 'NEW_YORK', 'ASIA', 'OVERLAP_LONDON_NY', or 'ALL'.
    """

    label: str
    min_distance_to_vwap_pips: float = 10.0
    max_distance_to_vwap_pips: float = 50.0
    require_rejection: bool = False
    session_name: str = "LONDON"
    direction_bias: TradeDirection | None = None
    max_spread_pips: float | None = None
    stop_pips: float = 10.0
    target_pips: float = 20.0
    max_label_bars: int = 50


class VWAPSweepResult(BaseModel, frozen=True):
    """Metrics from one parameter combination in a VWAP Fade sweep."""

    sweep_point: VWAPSweepPoint
    metrics: ReplayMetrics


class VWAPSweepComparison(BaseModel, frozen=True):
    """All results from a VWAP Fade parameter sweep, with ranking helpers.

    Methods return new sorted lists; the model itself is immutable.
    """

    results: list[VWAPSweepResult]

    def ranked_by_win_rate(self) -> list[VWAPSweepResult]:
        """Sort by win_rate_on_activated descending; None ranks last."""

        def _key(r: VWAPSweepResult) -> float:
            v = r.metrics.win_rate_on_activated
            return v if v is not None else -1.0

        return sorted(self.results, key=_key, reverse=True)

    def ranked_by_candidate_count(self) -> list[VWAPSweepResult]:
        """Sort by candidate_count descending."""
        return sorted(self.results, key=lambda r: r.metrics.candidate_count, reverse=True)

    def ranked_by_activation_rate(self) -> list[VWAPSweepResult]:
        """Sort by activation_rate descending; None ranks last."""

        def _key(r: VWAPSweepResult) -> float:
            v = r.metrics.activation_rate
            return v if v is not None else -1.0

        return sorted(self.results, key=_key, reverse=True)
