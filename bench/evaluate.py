"""Resumable evaluation for LoCoMo and LongMemEval with offline replay.

    python -m bench.evaluate --dataset locomo --data data/locomo10.json --dry-run

Stage caches are keyed independently: changing a context budget reuses memory,
changing a judge reuses predictions. Gold labels never enter memory creation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from sozograph import Passport, SozoGraph
from sozograph.batching import segment_interactions
from sozograph.ingest import coerce_to_interactions
from sozograph.providers import get_provider
from sozograph.retrieve import BM25
from sozograph.utils import sha256_json

from .locomo.judge import judge
from .locomo.runners import _ask
from .replay import code_hash, coverage, read_json, write_json


def _provider(spec, factory):
    return factory(spec)


def _ollama_factory(num_ctx=None, num_predict=None, repeat_penalty=None):
    """
    Build a provider_factory that threads Ollama-only options through.

    get_provider() takes a bare spec string here, with no way to reach
    OllamaProvider's num_ctx/num_predict/repeat_penalty fields. Ollama's own
    default context tops out at 4096 regardless of free VRAM, and its
    extraction schema arrays have no length limit, so a weak model can
    silently truncate a long segment or loop restating near-duplicate items
    without these set. Non-ollama specs pass through unchanged.
    """
    overrides = {k: v for k, v in
                 {"num_ctx": num_ctx, "num_predict": num_predict, "repeat_penalty": repeat_penalty}.items()
                 if v is not None}

    def factory(spec):
        kwargs = overrides if str(spec).startswith("ollama") else {}
        return get_provider(spec, **kwargs)

    return factory


def _usage(provider):
    return provider.usage.to_dict() if provider else {"calls": 0, "total_tokens": 0,
                                                     "prompt_tokens": 0, "completion_tokens": 0}


def _delta(after, before):
    return {key: after[key] - before.get(key, 0) for key in after}


def _lexical_context(turns, question, budget):
    interactions, _ = coerce_to_interactions(turns)
    chunks = segment_interactions(interactions, max_tokens=400)
    texts = [chunk.text() for chunk in chunks]
    scores = BM25(texts).score(question)
    context, ids = "SOURCE DATA", []
    for i in sorted(range(len(chunks)), key=lambda i: (-scores[i], i)):
        trial = context + "\n" + texts[i]
        if len(trial) <= budget:
            context = trial
            ids.extend(str(t.meta.get("parent_id") or t.id) for t in chunks[i].interactions)
    return (context if len(context) <= budget else ""), ids


def evaluate(conversations, *, out, extractor="openai:gpt-4o-mini", answerer=None,
             judge_model="openai:gpt-4o-mini", budget_chars=6000, segment_tokens=1500,
             retention="none", graph=False, include_sources=False, system="sozograph",
             stage="all", dataset_hash="", provider_factory=None, max_extra_calls=8):
    if system not in {"sozograph", "full_context", "lexical"}:
        raise ValueError("Unknown system")
    if stage not in {"build", "recall", "answer", "judge", "all"}:
        raise ValueError("Unknown stage")
    if budget_chars < 0:
        raise ValueError("budget_chars must be nonnegative")
    factory = provider_factory or get_provider
    answerer = answerer or extractor
    out = Path(out)
    code = code_hash()
    config = {"code_hash": code, "dataset_hash": dataset_hash, "extractor": extractor,
              "answerer": answerer, "judge": judge_model, "budget_chars": budget_chars,
              "segment_tokens": segment_tokens, "retention": retention, "graph": graph,
              "include_sources": include_sources, "system": system, "stage": stage,
              "max_extra_calls": max_extra_calls}
    run_id = sha256_json(config)[:24]
    run_path = out / "runs" / (run_id + ".json")
    report = {"config": config, "complete": False, "conversations": [],
              "judge_protocol": "SozoGraph equivalence judge; export hypotheses for official LongMemEval scoring"}
    write_json(run_path, report)
    memory_provider = qa_provider = judge_provider = None
    try:
        for conv in conversations:
            # Include only history in the construction key, never a gold label.
            normalized = [{k: v for k, v in t.items() if k in {"id", "text", "speaker", "ts", "date", "session"}}
                          for t in conv.turns]
            build_key = sha256_json([code, normalized, extractor, segment_tokens, retention, max_extra_calls])
            build_dir = out / "memory" / build_key
            memory_file = build_dir / "passport.json"
            build_state = read_json(build_dir / "state.json", {})
            passport = Passport.load(memory_file) if memory_file.exists() else None
            if system == "sozograph" and not build_state.get("complete"):
                if stage in {"recall", "answer", "judge"}:
                    raise ValueError("Complete memory cache required; run --stage build first")
                memory_provider = memory_provider or _provider(extractor, factory)
                before = _usage(memory_provider)
                started = time.perf_counter()

                def checkpoint(p, memory_file=memory_file, build_state=build_state,
                               memory_provider=memory_provider, before=before, started=started, build_dir=build_dir):
                    p.save(memory_file)
                    state = {"complete": False, "report": p.ingest_report,
                             "usage": {k: build_state.get("usage", {}).get(k, 0) + v
                                       for k, v in _delta(_usage(memory_provider), before).items()},
                             "seconds": build_state.get("seconds", 0) + time.perf_counter() - started}
                    write_json(build_dir / "state.json", state)

                write_json(build_dir / "input.json", normalized)
                passport = SozoGraph(memory_provider).ingest(
                    normalized, passport=passport, max_segment_tokens=segment_tokens,
                    retention=retention, max_extra_calls=max_extra_calls, checkpoint=checkpoint)
                build_state = read_json(build_dir / "state.json", {})
                build_state["complete"] = passport.ingest_report["complete"]
                write_json(build_dir / "state.json", build_state)
                if not build_state["complete"]:
                    raise RuntimeError("Memory construction is incomplete; checkpoint retained for retry")
            row = {"sample_id": conv.sample_id, "memory_key": build_key,
                   "memory_usage": build_state.get("usage", {}) if system == "sozograph" else {},
                   "memory_seconds": build_state.get("seconds", 0) if system == "sozograph" else 0,
                   "storage_bytes": len(passport.to_json(indent=None).encode("utf-8")) if passport else 0,
                   "questions": []}
            report["conversations"].append(row)
            if stage == "build":
                write_json(run_path, report)
                continue
            turn_sessions = {str(t.get("id")): str(t.get("session")) for t in normalized}
            for index, qa in enumerate(conv.qa):
                recall_key = sha256_json([build_key, system, qa.question, qa.question_date,
                                         budget_chars, graph, include_sources,
                                         passport.content_hash() if passport else None])
                recall_file = out / "recall" / (recall_key + ".json")
                recalled = read_json(recall_file)
                if recalled is None:
                    started = time.perf_counter()
                    if system == "sozograph":
                        result = passport.recall(query=qa.question, budget_chars=budget_chars,
                                                 graph=graph, include_sources=include_sources)
                        recalled = result.to_dict()
                        recalled["selected_turn_ids"] = sorted({tid for r in result.selected for tid in r.evidence_ids})
                        recalled["candidate_turn_ids"] = sorted({tid for r in result.selected + result.dropped
                                                               if r.reason not in {"subject filter", "time filter", "closed loop"}
                                                               for tid in r.evidence_ids})
                    elif system == "lexical":
                        context, ids = _lexical_context(normalized, qa.question, budget_chars)
                        recalled = {"context": context, "selected_turn_ids": ids,
                                    "candidate_turn_ids": list(turn_sessions), "used_chars": len(context)}
                    else:
                        context = "CONVERSATION:\n" + conv.as_text()
                        recalled = {"context": context, "selected_turn_ids": list(turn_sessions),
                                    "candidate_turn_ids": list(turn_sessions), "used_chars": len(context)}
                    recalled["seconds"] = time.perf_counter() - started
                    write_json(recall_file, recalled)
                # Question time is answer-side context, never a gold label.
                context = recalled["context"]
                if qa.question_date:
                    context = "Question date: " + qa.question_date + "\n" + context
                item = {"index": index, "question": qa.question, "gold": qa.answer,
                        "category": qa.category_name, "gold_evidence": qa.evidence,
                        "recall_key": recall_key, "recall_seconds": recalled["seconds"],
                        "context_chars": len(context),
                        "candidate_source_coverage": coverage(qa.evidence, recalled["candidate_turn_ids"]),
                        "rendered_source_coverage": coverage(qa.evidence, recalled["selected_turn_ids"]),
                        "rendered_session_coverage": coverage(qa.evidence_sessions,
                                                              [turn_sessions.get(t) for t in recalled["selected_turn_ids"]]),
                        "coverage_note": "Source membership, not semantic answer presence or verified entailment"}
                row["questions"].append(item)
                if stage == "recall":
                    write_json(run_path, report)
                    continue
                answer_key = sha256_json([recall_key, answerer, context, qa.question])
                answer_file = out / "answers" / (answer_key + ".json")
                answered = read_json(answer_file)
                if answered is None:
                    if stage == "judge":
                        raise ValueError("Answer cache required for judge-only replay")
                    qa_provider = qa_provider or _provider(answerer, factory)
                    before = _usage(qa_provider)
                    started = time.perf_counter()
                    prediction = _ask(qa_provider, context, qa.question)
                    answered = {"prediction": prediction, "usage": _delta(_usage(qa_provider), before),
                                "seconds": time.perf_counter() - started}
                    # Persist before any judge call, including the final question.
                    write_json(answer_file, answered)
                item.update(answered)
                if stage == "answer":
                    write_json(run_path, report)
                    continue
                judge_key = sha256_json([answer_key, judge_model, qa.answer, qa.category_name])
                judgment_file = out / "judgments" / (judge_key + ".json")
                judgment = read_json(judgment_file)
                if judgment is None:
                    judge_provider = judge_provider or _provider(judge_model, factory)
                    before = _usage(judge_provider)
                    verdict = judge(judge_provider, question=qa.question, gold=qa.answer,
                                    prediction=answered["prediction"])
                    judgment = {"correct": verdict.correct, "reason": verdict.reason,
                                "judge_usage": _delta(_usage(judge_provider), before)}
                    write_json(judgment_file, judgment)
                item.update(judgment)
                write_json(run_path, report)
        report["complete"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__}
        write_json(run_path, report)
        raise
    questions = [q for c in report["conversations"] for q in c["questions"]]
    marked = [q for q in questions if "correct" in q]
    categories = defaultdict(list)
    for q in marked:
        categories[q["category"]].append(q["correct"])
    report["metrics"] = {"judged": len(marked), "questions": len(questions),
                         "accuracy": sum(q["correct"] for q in marked) / len(marked) if marked else None,
                         "by_category": {k: sum(v) / len(v) for k, v in categories.items()},
                         "abstention_rate": sum(q.get("prediction") == "Not mentioned" for q in questions) / len(questions) if questions else None}
    write_json(run_path, report)
    # Official LongMemEval evaluator input. The report's local scores are not
    # represented as official scores; users can grade this export upstream.
    hypotheses = [{"question_id": c["sample_id"], "hypothesis": q["prediction"]}
                  for c in report["conversations"] for q in c["questions"] if "prediction" in q]
    export = out / "runs" / (run_id + ".hypotheses.jsonl")
    export.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in hypotheses), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["locomo", "longmemeval"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="bench/results/replay")
    parser.add_argument("--extractor", default="openai:gpt-4o-mini")
    parser.add_argument("--answerer")
    parser.add_argument("--judge", default="openai:gpt-4o-mini")
    parser.add_argument("--system", choices=["sozograph", "full_context", "lexical"], default="sozograph")
    parser.add_argument("--stage", choices=["build", "recall", "answer", "judge", "all"], default="all")
    parser.add_argument("--retention", choices=["none", "excerpts", "full"], default="none")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--include-sources", action="store_true")
    parser.add_argument("--budget-chars", type=int, default=6000)
    parser.add_argument("--segment-tokens", type=int, default=1500)
    parser.add_argument("--max-extra-calls", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-ctx", type=int, default=None,
                        help="Ollama context window, for --extractor/--answerer/--judge specs "
                             "that start with ollama (ignored otherwise)")
    parser.add_argument("--num-predict", type=int, default=None,
                        help="Ollama max completion tokens, same ollama-only scope as --num-ctx")
    parser.add_argument("--repeat-penalty", type=float, default=None,
                        help="Ollama repeat-penalty, same ollama-only scope as --num-ctx")
    args = parser.parse_args(argv)
    if args.dataset == "locomo":
        from .locomo.load import load_conversations
    else:
        from .longmemeval.load import load_conversations
    conversations = load_conversations(args.data, limit=args.limit, offset=args.offset)
    if args.dry_run:
        print(json.dumps({"conversations": len(conversations), "questions": sum(len(c.qa) for c in conversations),
                          "turns": sum(len(c.turns) for c in conversations), "calls": 0}))
        return 0
    report = evaluate(conversations, out=args.out, extractor=args.extractor, answerer=args.answerer,
                      judge_model=args.judge, system=args.system, stage=args.stage,
                      budget_chars=args.budget_chars, segment_tokens=args.segment_tokens,
                      retention=args.retention, graph=args.graph, include_sources=args.include_sources,
                      max_extra_calls=args.max_extra_calls,
                      provider_factory=_ollama_factory(args.num_ctx, args.num_predict, args.repeat_penalty),
                      dataset_hash=hashlib.sha256(Path(args.data).read_bytes()).hexdigest())
    print(json.dumps(report["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
