"""
research.edge_decay
────────────────────
Rolling-robustness / edge-decay analysis for strategy candidates.

Input  : trade-level return lists (from pattern/sequence backtests) or
         UnifiedCandidates + compact DataFrame.
Output : DecayReport per candidate — status (STABLE / IMPROVING /
         DECAYING / BROKEN) + continuous decay_score ∈ [-1, +1].
"""

from research.edge_decay.decay_report import (
    BROKEN_DD_THRESHOLD,
    DECAYING_REL_CHANGE,
    DecayReport,
    IMPROVING_REL_CHANGE,
    STABLE_PF_MIN,
    STATUS_BROKEN,
    STATUS_DECAYING,
    STATUS_IMPROVING,
    STATUS_INSUFFICIENT,
    STATUS_STABLE,
    build_report,
)
from research.edge_decay.rolling_metrics import (
    WindowMetrics,
    compute_rolling_metrics,
    compute_windows,
)

__all__ = [
    "WindowMetrics",
    "compute_rolling_metrics",
    "compute_windows",
    "DecayReport",
    "build_report",
    "STATUS_STABLE",
    "STATUS_IMPROVING",
    "STATUS_DECAYING",
    "STATUS_BROKEN",
    "STATUS_INSUFFICIENT",
    "BROKEN_DD_THRESHOLD",
    "IMPROVING_REL_CHANGE",
    "DECAYING_REL_CHANGE",
    "STABLE_PF_MIN",
]
