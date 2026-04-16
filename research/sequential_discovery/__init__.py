"""
research.sequential_discovery
──────────────────────────────
Research-only sequential (temporal) edge discovery.

Unlike the state-based pattern discovery, a *sequence* is an ORDERED
list of step events across consecutive bars:

    Sequence = tuple[tuple[col, bin_value], ...]   # length 2–4

  Example:
      (("range_compression_bin", "TRUE"),
       ("momentum_3_bin",        "POS"))

A sequence MATCHES at row `i` when:
    row[i]    == step 0
    row[i+1]  == step 1
    ...
    row[i+L-1] == step L-1

Entry is placed AFTER the final step, so the forward return we score
uses `forward_return_10[i + L - 1]` — consistent with the rest of the
research stack.

Public API
──────────
    Sequence               — ordered tuple alias
    SequenceResult         — result dataclass (incl. stability score)
    SequenceGenerator      — Apriori-style level-wise discovery
    discover_sequences()   — convenience wrapper
    evaluate_sequence()    — vectorised single-sequence scorer
"""

from research.sequential_discovery.sequence_evaluator import (
    SequenceResult,
    build_event_masks,
    evaluate_sequence,
    extend_end_mask,
)
from research.sequential_discovery.sequence_generator import (
    SequenceGenerator,
    discover_sequences,
)

Sequence = tuple  # tuple[tuple[str, str], ...]  — kept as a simple alias

__all__ = [
    "Sequence",
    "SequenceResult",
    "SequenceGenerator",
    "discover_sequences",
    "evaluate_sequence",
    "build_event_masks",
    "extend_end_mask",
]
