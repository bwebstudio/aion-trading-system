"""
aion/risk/allocator.py
───────────────────────
Risk Allocation v1 — main entry point.

Public API
----------
  evaluate(candidate, profile, state, instrument,
           stop_distance_points, target_distance_points=None)
      -> RiskDecision

Rule evaluation order (first failure short-circuits):
  1. check_equity                  — account is funded
  2. check_stop_distance           — stop is a valid positive number
  3. check_max_positions           — total concurrent positions within limit
  4. check_max_strategy_positions  — per-strategy positions within limit
  5. check_same_direction          — direction constraint (if configured)
  6. check_daily_risk              — daily risk budget not exceeded

If all rules pass, position size and risk amount are computed and
an approved RiskDecision is returned.

stop_distance_points and target_distance_points are always carried into the
RiskDecision (even on rejection) to support complete logging.
"""

from __future__ import annotations

from aion.core.models import InstrumentSpec
from aion.risk.models import PortfolioState, RiskDecision, RiskProfile
from aion.risk.rules import (
    check_daily_risk,
    check_equity,
    check_max_positions,
    check_max_strategy_positions,
    check_same_direction,
    check_stop_distance,
)
from aion.risk.sizing import compute_position_size, compute_risk_amount
from aion.strategies.models import CandidateSetup


def evaluate(
    candidate: CandidateSetup,
    profile: RiskProfile,
    state: PortfolioState,
    instrument: InstrumentSpec,
    stop_distance_points: float,
    target_distance_points: float | None = None,
) -> RiskDecision:
    """Evaluate a CandidateSetup against risk rules and compute position sizing.

    Parameters
    ----------
    candidate:
        The strategy signal to evaluate.  Provides setup_id, strategy_id,
        and direction for rule checks and decision metadata.
    profile:
        Account risk configuration (equity, limits, percentages).
    state:
        Current portfolio state BEFORE this trade is opened.  The caller is
        responsible for updating the state after a trade is executed.
    instrument:
        Instrument specification used for position sizing (point_value,
        min_lot, lot_step).
    stop_distance_points:
        Distance from entry to stop in instrument-native units.
        Use pips for forex (e.g. 10.0 = 10 pips for EURUSD).
        Use index points for indices (e.g. 10.0 = 10 points for US100).
        Must be positive.
    target_distance_points:
        Distance from entry to target, same units as stop.  Optional — not
        used in sizing but carried into RiskDecision for downstream use.

    Returns
    -------
    RiskDecision
        approved=True  with position_size and risk_amount when all rules pass.
        approved=False with reason_code and reason_text on the first failure.
    """

    def _reject(code: str, text: str) -> RiskDecision:
        return RiskDecision(
            approved=False,
            reason_code=code,
            reason_text=text,
            candidate_setup_id=candidate.setup_id,
            strategy_id=candidate.strategy_id,
            stop_distance_points=stop_distance_points,
            target_distance_points=target_distance_points,
        )

    # ── Rule 1: valid equity ──────────────────────────────────────────────────
    if rejection := check_equity(profile):
        return _reject(*rejection)

    # ── Rule 2: valid stop distance ───────────────────────────────────────────
    if rejection := check_stop_distance(stop_distance_points):
        return _reject(*rejection)

    # ── Rule 3: max concurrent positions ──────────────────────────────────────
    if rejection := check_max_positions(state, profile):
        return _reject(*rejection)

    # ── Rule 4: max positions per strategy ────────────────────────────────────
    if rejection := check_max_strategy_positions(state, profile, candidate.strategy_id):
        return _reject(*rejection)

    # ── Rule 5: direction constraint ──────────────────────────────────────────
    if rejection := check_same_direction(state, profile, candidate.direction):
        return _reject(*rejection)

    # ── Rule 6: daily risk budget ─────────────────────────────────────────────
    risk_amount = compute_risk_amount(profile)
    new_risk_pct = profile.max_risk_per_trade_pct
    if rejection := check_daily_risk(state, profile, new_risk_pct):
        return _reject(*rejection)

    # ── Compute position size ─────────────────────────────────────────────────
    position_size = compute_position_size(risk_amount, stop_distance_points, instrument)

    return RiskDecision(
        approved=True,
        reason_code="APPROVED",
        reason_text=(
            f"All risk checks passed. "
            f"Position size: {position_size} lot(s), "
            f"risk amount: {risk_amount:.2f} "
            f"({profile.max_risk_per_trade_pct:.1f}% of equity)."
        ),
        candidate_setup_id=candidate.setup_id,
        strategy_id=candidate.strategy_id,
        position_size=position_size,
        risk_amount=risk_amount,
        stop_distance_points=stop_distance_points,
        target_distance_points=target_distance_points,
    )
