"""
aion.replay.journal
────────────────────
CandidateJournal: mutable accumulator for replay records and labeled outcomes,
with simple JSONL persistence.

Design
───────
- Mutable by design — intended to be filled during a replay run.
- The records stored inside are immutable (frozen Pydantic models).
- JSONL format: one JSON object per line — easy to stream, grep, and load
  into pandas or any analytics tool.
- Accessors return copies so callers cannot mutate internal state.
- No analytics here — only storage and retrieval.

Typical usage
──────────────
    journal = CandidateJournal()

    for record in replay_result.records:
        journal.add_record(record)
    for label in replay_result.labeled_outcomes:
        journal.add_label(label)

    journal.save_records_jsonl(Path("runs/records.jsonl"))
    journal.save_labels_jsonl(Path("runs/labels.jsonl"))

    # Later, reload:
    records = CandidateJournal.load_records_jsonl(Path("runs/records.jsonl"))
"""

from __future__ import annotations

from pathlib import Path

from aion.replay.models import LabeledCandidateOutcome, ReplayEvaluationRecord
from aion.strategies.models import StrategyOutcome


class CandidateJournal:
    """
    Accumulates evaluation records and labeled outcomes during or after replay.

    All mutation is append-only.  Accessors return new lists (copies).
    """

    def __init__(self) -> None:
        self._records: list[ReplayEvaluationRecord] = []
        self._labels: list[LabeledCandidateOutcome] = []

    # ── Mutators ──────────────────────────────────────────────────────────────

    def add_record(self, record: ReplayEvaluationRecord) -> None:
        """Append one evaluation record."""
        self._records.append(record)

    def add_label(self, label: LabeledCandidateOutcome) -> None:
        """Append one labeled outcome."""
        self._labels.append(label)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def records(self) -> list[ReplayEvaluationRecord]:
        """All evaluation records in insertion order."""
        return list(self._records)

    def candidates(self) -> list[ReplayEvaluationRecord]:
        """Records where outcome == CANDIDATE."""
        return [
            r for r in self._records
            if r.evaluation_result.outcome == StrategyOutcome.CANDIDATE
        ]

    def no_trades(self) -> list[ReplayEvaluationRecord]:
        """Records where outcome == NO_TRADE."""
        return [
            r for r in self._records
            if r.evaluation_result.outcome == StrategyOutcome.NO_TRADE
        ]

    def insufficient_data(self) -> list[ReplayEvaluationRecord]:
        """Records where outcome == INSUFFICIENT_DATA."""
        return [
            r for r in self._records
            if r.evaluation_result.outcome == StrategyOutcome.INSUFFICIENT_DATA
        ]

    def labeled_outcomes(self) -> list[LabeledCandidateOutcome]:
        """All labeled outcomes in insertion order."""
        return list(self._labels)

    def __len__(self) -> int:
        """Total number of evaluation records."""
        return len(self._records)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_records_jsonl(self, path: Path) -> None:
        """
        Write all evaluation records to a JSONL file (one JSON object per line).

        Creates parent directories if they do not exist.
        Overwrites the file if it already exists.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for record in self._records:
                fh.write(record.model_dump_json())
                fh.write("\n")

    def save_labels_jsonl(self, path: Path) -> None:
        """Write all labeled outcomes to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for label in self._labels:
                fh.write(label.model_dump_json())
                fh.write("\n")

    @staticmethod
    def load_records_jsonl(path: Path) -> list[ReplayEvaluationRecord]:
        """
        Load evaluation records from a JSONL file.

        Returns an empty list if the file does not exist.
        """
        if not path.exists():
            return []
        records: list[ReplayEvaluationRecord] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(ReplayEvaluationRecord.model_validate_json(line))
        return records

    @staticmethod
    def load_labels_jsonl(path: Path) -> list[LabeledCandidateOutcome]:
        """
        Load labeled outcomes from a JSONL file.

        Returns an empty list if the file does not exist.
        """
        if not path.exists():
            return []
        labels: list[LabeledCandidateOutcome] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    labels.append(LabeledCandidateOutcome.model_validate_json(line))
        return labels
