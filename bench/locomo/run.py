"""
Run the LoCoMo benchmark.

    python -m bench.locomo.run --data data/locomo10.json

Defaults match LightMem's published setup: GPT-4o-mini as both backbone and
judge, the four non-adversarial question categories, per-conversation figures.
"""
from __future__ import annotations

import argparse
import sys

from sozograph.providers import get_provider

from .judge import judge
from .load import DEFAULT_CATEGORIES, describe, load_conversations
from .metrics import aggregate, render_table, save
from .runners import RUNNERS


def _progress(label: str, index: int, total: int) -> None:
    print(f"  [{index}/{total}] {label}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.locomo.run", description="Run SozoGraph against LoCoMo."
    )
    parser.add_argument("--data", default="data/locomo10.json",
                        help="Path to locomo10.json")
    parser.add_argument("--provider", default="openai:gpt-4o-mini",
                        help="Backbone, as provider[:model]")
    parser.add_argument("--judge", default="openai:gpt-4o-mini",
                        help="Judge, as provider[:model]")
    parser.add_argument("--systems", default="sozograph",
                        help="Comma-separated: sozograph, full_context")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N conversations")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first N conversations (for daily quota-split batches)")
    parser.add_argument("--budget-chars", type=int, default=6000,
                        help="Context budget per question")
    parser.add_argument("--segment-tokens", type=int, default=1500,
                        help="Target tokens per extraction segment")
    parser.add_argument("--compact", action="store_true",
                        help="Run semantic reconciliation after ingestion")
    parser.add_argument("--categories", default=",".join(str(c) for c in DEFAULT_CATEGORIES),
                        help="Question categories to include")
    parser.add_argument("--out", default="bench/results", help="Where to write results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Describe the dataset and exit without calling anything")
    args = parser.parse_args(argv)

    categories = tuple(int(c) for c in args.categories.split(",") if c.strip())
    try:
        conversations = load_conversations(
            args.data, categories=categories, limit=args.limit, offset=args.offset
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    summary = describe(conversations)
    print("LoCoMo dataset")
    for key, value in summary.items():
        print(f"  {key:34s} {value}")
    print()

    if args.dry_run:
        print("Dry run: no model was called.")
        return 0

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in systems if s not in RUNNERS]
    if unknown:
        print(f"Unknown system(s): {unknown}. Choose from {list(RUNNERS)}", file=sys.stderr)
        return 2

    judge_provider = get_provider(args.judge)
    results, all_metrics = {}, []

    def _save_now(*, note: str | None = None):
        config = {
            "provider": args.provider,
            "judge": args.judge,
            "categories": list(categories),
            "budget_chars": args.budget_chars,
            "segment_tokens": args.segment_tokens,
            "compact": args.compact,
            "offset": args.offset,
            "conversations": len(conversations),
            "judge_usage": judge_provider.usage.to_dict(),
        }
        if note:
            config["note"] = note
        return save(all_metrics, results, args.out, config=config)

    # A quota-constrained provider (a free-tier daily cap, say) can die mid-run
    # after real, already-paid-for API calls. Losing that work on every crash
    # is worse than the crash itself, so whatever finished is saved either way.
    try:
        for system in systems:
            print(f"Running {system} on {len(conversations)} conversation(s) "
                  f"with {args.provider}")
            runner = RUNNERS[system]
            runs, verdicts = [], []
            try:
                for index, conversation in enumerate(conversations, start=1):
                    _progress(f"{conversation.sample_id} "
                              f"({len(conversation.turns)} turns, {len(conversation.qa)} questions)",
                              index, len(conversations))
                    if system == "sozograph":
                        result = runner(
                            conversation,
                            model=args.provider,
                            budget_chars=args.budget_chars,
                            max_segment_tokens=args.segment_tokens,
                            compact_after=args.compact,
                        )
                    else:
                        result = runner(conversation, model=args.provider)

                    marks = [
                        judge(judge_provider, question=a.question, gold=a.gold,
                              prediction=a.prediction).correct
                        for a in result.answers
                    ]
                    runs.append(result)
                    verdicts.append(marks)

                    hits = sum(marks)
                    print(f"      {hits}/{len(marks)} correct, "
                          f"{result.total_tokens:,} tokens, {result.total_calls} calls")
            finally:
                if runs:
                    results[system] = runs
                    all_metrics.append(aggregate(system, runs, verdicts))
    except Exception:
        path = _save_now(note="partial: run raised before completing; see stderr for the cause")
        done = sum(len(r) for r in results.values())
        print(f"\nCrashed partway through -- {done} system-conversation run(s) "
              f"already completed were saved to {path}", file=sys.stderr)
        raise

    print()
    print(render_table(all_metrics))
    print()

    path = _save_now()
    print(f"Results written to {path}")

    for metrics in all_metrics:
        print(f"\n{metrics.system} accuracy by category:")
        for name, value in metrics.category_accuracy().items():
            print(f"  {name:16s} {value:6.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
