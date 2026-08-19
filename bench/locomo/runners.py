"""
The two systems under test.

`full_context` is the honest upper bound: put the whole conversation in the
prompt every time. It scores well and costs enormously, which is the tradeoff
every memory system exists to improve on.

`sozograph` builds a passport once, then answers each question from a
query-selected slice of it. Memory tokens and QA tokens are measured separately
so the comparison against LightMem's published table is like for like.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sozograph import Passport, SozoGraph
from sozograph.providers import LLMProvider, get_provider
from sozograph.providers.base import Usage

from .load import Conversation

ANSWER_SYSTEM_PROMPT = """
You answer questions about a conversation using only the information provided.

Rules:
- Answer in as few words as possible. A date, a name, or a short phrase.
- Do not explain your reasoning and do not restate the question.
- If the information is genuinely not present, say "Not mentioned".
- Prefer the most recent information when something changed over time.
""".strip()

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

ANSWER_USER_TEMPLATE = """
{context}

QUESTION: {question}

Answer in as few words as possible.
""".strip()


@dataclass
class Answer:
    question: str
    gold: str
    prediction: str
    category: int


@dataclass
class RunResult:
    system: str
    sample_id: str
    answers: list[Answer] = field(default_factory=list)
    memory_usage: Usage = field(default_factory=Usage)
    qa_usage: Usage = field(default_factory=Usage)
    memory_seconds: float = 0.0
    qa_seconds: float = 0.0
    passport_tokens: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.memory_usage.total_tokens + self.qa_usage.total_tokens

    @property
    def total_calls(self) -> int:
        return self.memory_usage.calls + self.qa_usage.calls


def _ask(provider: LLMProvider, context: str, question: str) -> str:
    payload = provider.complete_json(
        system=ANSWER_SYSTEM_PROMPT,
        user=ANSWER_USER_TEMPLATE.format(context=context, question=question),
        schema=ANSWER_SCHEMA,
        temperature=0.0,
    )
    return str(payload.get("answer") or "").strip()


def run_full_context(
    conversation: Conversation, *, model: str, provider_kwargs: dict[str, Any] | None = None
) -> RunResult:
    """Baseline: the entire conversation in every prompt."""
    result = RunResult(system="full_context", sample_id=conversation.sample_id)
    provider = get_provider(model, **(provider_kwargs or {}))
    context = "CONVERSATION:\n" + conversation.as_text()

    started = time.perf_counter()
    for qa in conversation.qa:
        result.answers.append(
            Answer(qa.question, qa.answer, _ask(provider, context, qa.question), qa.category)
        )
    result.qa_seconds = time.perf_counter() - started
    result.qa_usage = provider.usage
    result.notes["context_tokens"] = conversation.estimated_tokens()
    return result


def run_sozograph(
    conversation: Conversation,
    *,
    model: str,
    budget_chars: int = 6000,
    max_segment_tokens: int = 1500,
    compact_after: bool = False,
    provider_kwargs: dict[str, Any] | None = None,
) -> RunResult:
    """Build a passport once, then answer from a query-selected slice."""
    result = RunResult(system="sozograph", sample_id=conversation.sample_id)

    # Separate providers so memory-construction tokens and question-answering
    # tokens are never conflated. LightMem's table splits them; so does this.
    memory_provider = get_provider(model, **(provider_kwargs or {}))
    qa_provider = get_provider(model, **(provider_kwargs or {}))

    started = time.perf_counter()
    graph = SozoGraph(provider=memory_provider)
    passport: Passport = graph.ingest(
        conversation.turns,
        meta={"user_key": conversation.sample_id},
        max_segment_tokens=max_segment_tokens,
    )
    if compact_after:
        from sozograph import compact

        compact(passport, memory_provider)
    result.memory_seconds = time.perf_counter() - started
    result.memory_usage = memory_provider.usage
    result.passport_tokens = passport.token_estimate()
    result.notes.update({
        "segments": len(passport.stats),
        "facts": len(passport.facts),
        "prefs": len(passport.prefs),
        "episodes": len(passport.episodes),
        "entities": len(passport.entities),
        "contradictions": len(passport.contradictions),
    })

    started = time.perf_counter()
    for qa in conversation.qa:
        context = passport.context(
            query=qa.question, budget_chars=budget_chars, header="WHAT I REMEMBER"
        )
        result.answers.append(
            Answer(qa.question, qa.answer, _ask(qa_provider, context, qa.question), qa.category)
        )
    result.qa_seconds = time.perf_counter() - started
    result.qa_usage = qa_provider.usage
    return result


RUNNERS = {
    "sozograph": run_sozograph,
    "full_context": run_full_context,
}
