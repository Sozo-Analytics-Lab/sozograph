"""
Aggregate run results into the table the README publishes.

Column names match LightMem's Table 3 so the two can sit side by side: accuracy,
memory tokens, QA tokens, total tokens, API calls, runtime. Per-conversation
averages, because that is how the published figures are reported.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .load import CATEGORY_NAMES
from .runners import RunResult

#: Published figures for context. These are LightMem's own numbers from Table 3
#: of arXiv:2510.18866 (GPT-4o-mini backbone, GPT-4o-mini judge), per
#: conversation. We do not re-run their system; installing it costs conda plus
#: several gigabytes of model weights. Their row is cited, ours is measured.
PUBLISHED = {
    "LightMem (0.8, 768)": {"accuracy": 72.99, "memory_tokens_k": 85.19, "api_calls": 29.83},
    "A-MEM": {"accuracy": None, "memory_tokens_k": 1149.43, "api_calls": 1175.47},
    "Mem0": {"accuracy": 36.49, "memory_tokens_k": 1693.39, "api_calls": 1602.20},
}


@dataclass
class Metrics:
    system: str
    conversations: int = 0
    questions: int = 0
    correct: int = 0
    memory_tokens: int = 0
    qa_tokens: int = 0
    memory_calls: int = 0
    qa_calls: int = 0
    memory_seconds: float = 0.0
    qa_seconds: float = 0.0
    passport_tokens: int = 0
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / self.questions if self.questions else 0.0

    @property
    def total_tokens(self) -> int:
        return self.memory_tokens + self.qa_tokens

    @property
    def total_calls(self) -> int:
        return self.memory_calls + self.qa_calls

    def per_conversation(self, value: float) -> float:
        return value / self.conversations if self.conversations else 0.0

    def category_accuracy(self) -> dict[str, float]:
        return {
            name: 100.0 * counts["correct"] / counts["total"] if counts["total"] else 0.0
            for name, counts in sorted(self.by_category.items())
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "conversations": self.conversations,
            "questions": self.questions,
            "accuracy_pct": round(self.accuracy, 2),
            "memory_tokens_per_conversation": round(self.per_conversation(self.memory_tokens)),
            "qa_tokens_per_conversation": round(self.per_conversation(self.qa_tokens)),
            "total_tokens_per_conversation": round(self.per_conversation(self.total_tokens)),
            "memory_calls_per_conversation": round(self.per_conversation(self.memory_calls), 2),
            "qa_calls_per_conversation": round(self.per_conversation(self.qa_calls), 2),
            "total_calls_per_conversation": round(self.per_conversation(self.total_calls), 2),
            "memory_seconds_per_conversation": round(self.per_conversation(self.memory_seconds), 1),
            "qa_seconds_per_conversation": round(self.per_conversation(self.qa_seconds), 1),
            "passport_tokens_mean": round(self.per_conversation(self.passport_tokens)),
            "accuracy_by_category": {k: round(v, 2) for k, v in self.category_accuracy().items()},
        }


def aggregate(system: str, results: list[RunResult], verdicts: list[list[bool]]) -> Metrics:
    metrics = Metrics(system=system, conversations=len(results))
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for result, marks in zip(results, verdicts, strict=True):
        metrics.memory_tokens += result.memory_usage.total_tokens
        metrics.qa_tokens += result.qa_usage.total_tokens
        metrics.memory_calls += result.memory_usage.calls
        metrics.qa_calls += result.qa_usage.calls
        metrics.memory_seconds += result.memory_seconds
        metrics.qa_seconds += result.qa_seconds
        metrics.passport_tokens += result.passport_tokens

        for answer, ok in zip(result.answers, marks, strict=True):
            metrics.questions += 1
            metrics.correct += int(ok)
            name = CATEGORY_NAMES.get(answer.category, f"category_{answer.category}")
            buckets[name]["total"] += 1
            buckets[name]["correct"] += int(ok)

    metrics.by_category = dict(buckets)
    return metrics


def render_table(all_metrics: list[Metrics], *, include_published: bool = True) -> str:
    """The comparison table, as markdown."""
    header = (
        "| System | Accuracy | Memory tokens | QA tokens | Total tokens | API calls | Runtime |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for m in all_metrics:
        rows.append(
            f"| **{m.system}** (measured) | {m.accuracy:.2f}% "
            f"| {m.per_conversation(m.memory_tokens) / 1000:.2f}k "
            f"| {m.per_conversation(m.qa_tokens) / 1000:.2f}k "
            f"| {m.per_conversation(m.total_tokens) / 1000:.2f}k "
            f"| {m.per_conversation(m.total_calls):.2f} "
            f"| {m.per_conversation(m.memory_seconds + m.qa_seconds):.0f}s |"
        )

    if include_published:
        for name, row in PUBLISHED.items():
            accuracy = f"{row['accuracy']:.2f}%" if row["accuracy"] is not None else "n/a"
            rows.append(
                f"| {name} (published) | {accuracy} "
                f"| {row['memory_tokens_k']:.2f}k | n/a | n/a "
                f"| {row['api_calls']:.2f} | n/a |"
            )

    footnote = (
        "\nPublished rows are LightMem's Table 3 (arXiv:2510.18866), GPT-4o-mini "
        "backbone and judge, per conversation. They were not re-run here; only "
        "the measured rows come from this harness."
    )
    return "\n".join([header, *rows]) + ("\n" + footnote if include_published else "")


def save(all_metrics: list[Metrics], results: dict[str, list[RunResult]],
         out_dir: str | Path, *, config: dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"locomo_{stamp}.json"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "published_reference": PUBLISHED,
        "metrics": [m.to_dict() for m in all_metrics],
        "per_conversation": {
            system: [
                {
                    "sample_id": r.sample_id,
                    "memory_tokens": r.memory_usage.to_dict(),
                    "qa_tokens": r.qa_usage.to_dict(),
                    "memory_seconds": round(r.memory_seconds, 2),
                    "qa_seconds": round(r.qa_seconds, 2),
                    "passport_tokens": r.passport_tokens,
                    "notes": r.notes,
                }
                for r in runs
            ]
            for system, runs in results.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
