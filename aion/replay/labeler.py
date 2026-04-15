"""
aion.replay.labeler
────────────────────
Forward-looking candidate labeler.

Produces a LabeledCandidateOutcome for a CandidateSetup given a list of
future MarketBar objects and a LabelConfig.

What it simulates
──────────────────
1. Entry activation   — first bar where price crosses entry_reference level.
2. Stop / target scan — bar by bar after activation: first touch wins.
3. MFE / MAE          — measured in pips from entry_reference across all
                        post-activation bars until resolution.

What it does NOT simulate
──────────────────────────
- Slippage or fills (entry assumed at entry_reference exactly).
- Partial fills or liquidity constraints.
- Commissions or financing costs.
- Multiple concurrent positions.

Entry logic
────────────
  LONG  — activated when bar.high >= entry_reference
  SHORT — activated when bar.low  <= entry_reference

Stop / target logic
────────────────────
  LONG:  stop when bar.low  <= stop_price
         target when bar.high >= target_price
  SHORT: stop when bar.high >= stop_price
         target when bar.low  <= target_price

Tie-breaking rule (same-bar stop + target)
───────────────────────────────────────────
If a bar simultaneously satisfies both stop and target conditions (a wide
bar spanning both levels), LOSS is returned — the conservative assumption
that the stop was touched first.
"""

from __future__ import annotations

from aion.core.models import MarketBar
from aion.strategies.models import CandidateSetup
from aion.replay.models import LabelConfig, LabelOutcome, LabeledCandidateOutcome


def label_candidate(
    candidate: CandidateSetup,
    future_bars: list[MarketBar],
    label_config: LabelConfig,
) -> LabeledCandidateOutcome:
    """
    Attach a forward-looking label to a CandidateSetup.

    Parameters
    ----------
    candidate:
        The setup to label.  entry_reference, direction, symbol, and
        timestamp_utc are used.
    future_bars:
        Bars *after* the bar that generated the candidate, in chronological
        order.  Typically built from subsequent snapshots' latest_bar.
        At most label_config.max_bars bars are inspected.
    label_config:
        Defines stop/target distances in pips and instrument parameters.

    Returns
    -------
    LabeledCandidateOutcome
        Fully populated; outcome is one of WIN / LOSS / TIMEOUT /
        ENTRY_NOT_ACTIVATED.
    """
    pip_size = label_config.tick_size * label_config.pip_multiplier
    entry = candidate.entry_reference
    is_long = candidate.is_long

    # ── Compute absolute stop and target price levels ─────────────────────────
    stop_dist = label_config.stop_pips * pip_size
    tgt_dist = label_config.target_pips * pip_size

    if is_long:
        stop_price = entry - stop_dist
        target_price = entry + tgt_dist
    else:
        stop_price = entry + stop_dist
        target_price = entry - tgt_dist

    # ── Limit look-ahead to max_bars ─────────────────────────────────────────
    bars = future_bars[: label_config.max_bars]

    if not bars:
        return _not_activated(candidate, stop_price, target_price)

    # ── Phase 1: find entry activation ───────────────────────────────────────
    activation_index: int | None = None
    for idx, bar in enumerate(bars):
        if is_long and bar.high >= entry:
            activation_index = idx
            break
        if not is_long and bar.low <= entry:
            activation_index = idx
            break

    if activation_index is None:
        return _not_activated(candidate, stop_price, target_price)

    bars_to_entry = activation_index
    post_activation = bars[activation_index:]

    # ── Phase 2: scan for stop / target and track MFE / MAE ──────────────────
    best_favorable: float = 0.0  # running MFE numerator (price units)
    worst_adverse: float = 0.0   # running MAE numerator (price units, always >= 0)

    outcome = LabelOutcome.TIMEOUT
    bars_to_resolution: int | None = None

    for offset, bar in enumerate(post_activation):
        if is_long:
            favorable = bar.high - entry
            adverse = entry - bar.low
        else:
            favorable = entry - bar.low
            adverse = bar.high - entry

        best_favorable = max(best_favorable, favorable)
        worst_adverse = max(worst_adverse, adverse)

        # Conservative tie-break: check stop before target on the same bar.
        stop_hit = (bar.low <= stop_price) if is_long else (bar.high >= stop_price)
        target_hit = (bar.high >= target_price) if is_long else (bar.low <= target_price)

        if stop_hit:
            outcome = LabelOutcome.LOSS
            bars_to_resolution = offset
            break

        if target_hit:
            outcome = LabelOutcome.WIN
            bars_to_resolution = offset
            break

    mfe_pips = round(best_favorable / pip_size, 4) if pip_size > 0 else 0.0
    mae_pips = round(worst_adverse / pip_size, 4) if pip_size > 0 else 0.0

    if outcome == LabelOutcome.WIN:
        pnl_pips: float | None = label_config.target_pips
    elif outcome == LabelOutcome.LOSS:
        pnl_pips = -label_config.stop_pips
    else:
        pnl_pips = None  # TIMEOUT — unresolved within look-ahead window

    return LabeledCandidateOutcome(
        setup_id=candidate.setup_id,
        symbol=candidate.symbol,
        timestamp_utc=candidate.timestamp_utc,
        direction=candidate.direction,
        entry_reference=entry,
        stop_price=stop_price,
        target_price=target_price,
        outcome=outcome,
        entry_activated=True,
        bars_to_entry=bars_to_entry,
        bars_to_resolution=bars_to_resolution,
        mfe_pips=mfe_pips,
        mae_pips=mae_pips,
        pnl_pips=pnl_pips,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _not_activated(
    candidate: CandidateSetup,
    stop_price: float,
    target_price: float,
) -> LabeledCandidateOutcome:
    return LabeledCandidateOutcome(
        setup_id=candidate.setup_id,
        symbol=candidate.symbol,
        timestamp_utc=candidate.timestamp_utc,
        direction=candidate.direction,
        entry_reference=candidate.entry_reference,
        stop_price=stop_price,
        target_price=target_price,
        outcome=LabelOutcome.ENTRY_NOT_ACTIVATED,
        entry_activated=False,
        bars_to_entry=None,
        bars_to_resolution=None,
        mfe_pips=None,
        mae_pips=None,
        pnl_pips=None,
    )
