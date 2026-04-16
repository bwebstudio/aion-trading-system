"""
research.sequential_strategies.backtest_sequential_strategy
────────────────────────────────────────────────────────────
Bar-replay backtest for a SequentialStrategyCandidate.

Replay model
────────────
For each row j where the candidate's ordered sequence COMPLETES
(steps 0..L-1 matched rows j-L+1..j), open a position at the OPEN of
bar j+1.  Intrabar high/low drives the ATR stop / ATR take-profit
checks; on timeout the position closes at the timeout bar's close.

Single position at a time — a new sequence completion is only acted on
after the prior position has closed (no overlap, no pyramiding).

The entry-mask is produced by the end-mask recurrence from
`research.sequential_discovery.sequence_evaluator`:

    mask_0  = rows where step 0 is True
    mask_L  = shift_right(mask_{L-1}, 1) AND rows where step L is True

The result is reported as a `BacktestReport` (reused from
pattern_strategies) — same schema, same summary line, so downstream
tooling treats pattern and sequence strategies uniformly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from research.pattern_strategies.backtest_pattern_strategy import (
    BacktestReport,
    Trade,
    _build_report,
    _rule_multiplier,
)
from research.pattern_strategies.strategy_candidate import StrategyCandidate
from research.sequential_discovery.sequence_evaluator import (
    build_event_masks,
    extend_end_mask,
)
from research.sequential_strategies.strategy_candidate import (
    SequentialStrategyCandidate,
)

if TYPE_CHECKING:
    import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# End-mask builder
# ─────────────────────────────────────────────────────────────────────────────


def _sequence_end_mask(
    df: "pd.DataFrame",
    candidate: SequentialStrategyCandidate,
) -> np.ndarray:
    """
    Return a boolean array the length of `df` where mask[j] is True iff
    the candidate's ordered sequence completed at row j.

    If any step references a column or value absent from `df`, the mask
    is all-False (the sequence can never fire).
    """
    n = len(df)
    if not candidate.sequence_key:
        return np.zeros(n, dtype=bool)

    cols_needed = tuple({col for col, _ in candidate.sequence_key})
    event_masks = build_event_masks(df, cols_needed, min_support=1)

    first = candidate.sequence_key[0]
    first_mask = event_masks.get(first)
    if first_mask is None:
        return np.zeros(n, dtype=bool)

    end_mask = first_mask.copy()
    for step in candidate.sequence_key[1:]:
        sm = event_masks.get(step)
        if sm is None:
            return np.zeros(n, dtype=bool)
        end_mask = extend_end_mask(end_mask, sm)
    return end_mask


# ─────────────────────────────────────────────────────────────────────────────
# Replay
# ─────────────────────────────────────────────────────────────────────────────


def backtest_sequential_candidate(
    df: "pd.DataFrame",
    candidate: SequentialStrategyCandidate,
) -> BacktestReport:
    """
    Replay `candidate` over `df`.  The df must have `open, high, low,
    close, atr_14` plus the categorical bin columns referenced by the
    candidate's sequence.
    """
    required = {"open", "high", "low", "close", "atr_14"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {sorted(missing)}"
        )

    stop_mult = _rule_multiplier(candidate.stop_rule)
    tp_rule = candidate.exit_rule.get("take_profit", {})
    tp_mult = _rule_multiplier(tp_rule)
    max_hold = int(candidate.exit_rule.get("max_hold_bars", 20))
    direction = candidate.direction
    long_ = direction == "LONG"

    end_mask = _sequence_end_mask(df, candidate)

    opens = df["open"].to_numpy(dtype=np.float64, copy=False)
    highs = df["high"].to_numpy(dtype=np.float64, copy=False)
    lows = df["low"].to_numpy(dtype=np.float64, copy=False)
    closes = df["close"].to_numpy(dtype=np.float64, copy=False)
    atrs = df["atr_14"].to_numpy(dtype=np.float64, copy=False)

    n = len(df)
    trades: list[Trade] = []
    i = 0

    while i < n - 1:
        # Entry fires on the bar AFTER the sequence completed at row i.
        if not end_mask[i]:
            i += 1
            continue

        entry_bar = i + 1
        if entry_bar >= n:
            break
        entry_price = opens[entry_bar]
        atr = atrs[i]
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry_price):
            i += 1
            continue

        if long_:
            stop = entry_price - stop_mult * atr
            tp = entry_price + tp_mult * atr
        else:
            stop = entry_price + stop_mult * atr
            tp = entry_price - tp_mult * atr

        exit_bar: int | None = None
        exit_price: float | None = None
        exit_reason = "TIMEOUT"
        last_allowed = min(entry_bar + max_hold - 1, n - 1)

        for j in range(entry_bar, last_allowed + 1):
            hi, lo = highs[j], lows[j]
            if long_:
                stop_hit = lo <= stop
                tp_hit = hi >= tp
            else:
                stop_hit = hi >= stop
                tp_hit = lo <= tp
            # Stop wins ties (conservative).
            if stop_hit:
                exit_bar, exit_price, exit_reason = j, stop, "STOP"
                break
            if tp_hit:
                exit_bar, exit_price, exit_reason = j, tp, "TP"
                break

        if exit_bar is None:
            exit_bar = last_allowed
            exit_price = closes[exit_bar]
            exit_reason = "TIMEOUT"

        if long_:
            ret = (exit_price - entry_price) / entry_price
        else:
            ret = (entry_price - exit_price) / entry_price

        trades.append(
            Trade(
                entry_bar=entry_bar,
                exit_bar=exit_bar,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                direction=direction,
                stop_price=float(stop),
                tp_price=float(tp),
                return_pct=float(ret),
                exit_reason=exit_reason,
            )
        )
        # No overlapping positions.
        i = exit_bar + 1

    # Wrap the sequential candidate in a zero-width StrategyCandidate shim
    # so we can reuse the existing _build_report helper unchanged.
    shim = StrategyCandidate(
        pattern_key=(("_seq_", candidate.name),),
        direction=direction,
        entry_rule={},
        stop_rule={},
        exit_rule={},
        expected_edge={},
        name=candidate.name,
    )
    return _build_report(shim, trades)


__all__ = [
    "backtest_sequential_candidate",
    "BacktestReport",
]
