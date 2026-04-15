"""
aion/risk/models.py
────────────────────
Domain models for Risk Allocation v1.

Three core models:
  RiskProfile    — account-level risk configuration (limits, percentages).
  PortfolioState — portfolio snapshot at the moment of evaluating a new trade.
  RiskDecision   — output of the allocator (verdict + sizing details).

All models are immutable (frozen=True).  RiskDecision carries both the
approval verdict and the computed sizing so downstream modules (execution,
journal, dashboard) consume a single object.

Design rules:
  - No business logic here — only data + field-level validation.
  - All monetary amounts are in the account's profit currency (e.g. USD).
  - Percentages are 0–100, NOT 0–1.  1.0 means 1%, not 100%.
  - reason_text is intentionally non-technical: it will be displayed on a
    non-technical dashboard, not consumed by algorithms.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RiskProfile(BaseModel, frozen=True):
    """Risk configuration for one trading account.

    Instantiate once at startup from a config file or user settings.
    All percentage fields use the 0–100 scale (1.0 = 1%).

    Example (conservative setup):
        RiskProfile(
            account_equity=10_000.0,
            max_risk_per_trade_pct=1.0,
            max_daily_risk_pct=2.0,
            max_concurrent_positions=3,
            max_positions_per_strategy=1,
            allow_same_direction_multiple=False,
        )
    """

    account_equity: float
    """Current account equity in the profit currency (e.g. USD).  Must be > 0."""

    max_risk_per_trade_pct: float = 1.0
    """Maximum risk per individual trade as a percentage of equity.
    E.g. 1.0 means risk at most 1% of the account on any single trade."""

    max_daily_risk_pct: float = 2.0
    """Maximum cumulative risk allowed across all trades in a single day.
    E.g. 2.0 means stop opening new trades once 2% of equity is at risk today."""

    max_concurrent_positions: int = 3
    """Maximum number of simultaneously open positions across all strategies."""

    max_positions_per_strategy: int = 2
    """Maximum open positions allowed for any single strategy at the same time."""

    allow_same_direction_multiple: bool = False
    """If False, a second position in the same direction (e.g. two LONGs) is
    rejected, regardless of strategy.  Set True to allow stacking in one direction."""

    @field_validator("account_equity")
    @classmethod
    def equity_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"account_equity must be positive, got {v}.")
        return v

    @field_validator("max_risk_per_trade_pct", "max_daily_risk_pct")
    @classmethod
    def pct_in_range(cls, v: float) -> float:
        if v <= 0 or v > 100:
            raise ValueError(
                f"Risk percentages must be between 0 (exclusive) and 100, got {v}."
            )
        return v

    @field_validator("max_concurrent_positions", "max_positions_per_strategy")
    @classmethod
    def limits_must_be_at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Position limits must be at least 1, got {v}.")
        return v


class PortfolioState(BaseModel, frozen=True):
    """Current state of the portfolio at the moment of evaluating a new trade.

    This is a read-only snapshot taken BEFORE the candidate trade is opened.
    The allocator reads it to apply limits without mutating state — the caller
    is responsible for updating the state after a trade is executed.

    Constructing an empty PortfolioState() represents a fresh trading day
    with no open positions and no daily risk consumed.
    """

    open_positions_count: int = 0
    """Total number of currently open positions across all strategies."""

    open_positions_by_strategy: dict[str, int] = Field(default_factory=dict)
    """strategy_id -> number of open positions for that strategy."""

    open_positions_by_direction: dict[str, int] = Field(default_factory=dict)
    """Direction -> number of open positions in that direction.
    Keys are TradeDirection.value strings: 'LONG' or 'SHORT'."""

    daily_realized_pnl: float = 0.0
    """Realized profit/loss for the current trading day (account currency).
    Positive = profit, negative = loss."""

    daily_unrealized_pnl: float = 0.0
    """Unrealized profit/loss across all currently open positions."""

    daily_risk_used_pct: float = 0.0
    """Fraction of the daily risk budget already committed, as a percentage of equity.
    E.g. 1.5 means 1.5% of equity is currently at risk in open positions."""


class RiskDecision(BaseModel, frozen=True):
    """Output of the risk allocator for one candidate trade setup.

    When approved=True:
      - position_size and risk_amount are populated.
      - The trade can be forwarded to the execution module.

    When approved=False:
      - reason_code identifies which rule blocked the trade.
      - reason_text provides a human-readable explanation (non-technical,
        suitable for a user-facing dashboard).
      - position_size and risk_amount are None.

    stop_distance_points and target_distance_points are always carried through
    (even on rejection) so the caller can log the full context.
    """

    approved: bool
    """True if all risk rules passed and sizing was computed."""

    reason_code: str
    """
    Short, stable identifier for the decision outcome.
      'APPROVED'                      — all checks passed
      'INVALID_EQUITY'                — account equity is zero or negative
      'INVALID_STOP_DISTANCE'         — stop distance is zero or negative
      'MAX_POSITIONS_REACHED'         — too many concurrent positions
      'MAX_STRATEGY_POSITIONS_REACHED'— too many positions for this strategy
      'SAME_DIRECTION_NOT_ALLOWED'    — direction constraint violated
      'MAX_DAILY_RISK_REACHED'        — daily risk budget exhausted
    """

    reason_text: str
    """Human-readable explanation, clear enough for a non-technical dashboard."""

    candidate_setup_id: str
    """Matches CandidateSetup.setup_id — for tracing and logging."""

    strategy_id: str
    """Matches CandidateSetup.strategy_id."""

    position_size: float | None = None
    """Computed position size in lots.  None when the trade is rejected."""

    risk_amount: float | None = None
    """Monetary amount at risk on this trade (account currency).
    None when the trade is rejected."""

    stop_distance_points: float | None = None
    """Stop distance used for sizing (pips for forex, index points for indices)."""

    target_distance_points: float | None = None
    """Target distance (pips or points).  None if not provided by the caller."""
