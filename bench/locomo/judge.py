"""
LLM-as-judge scoring, following LightMem's protocol.

Accuracy is the proportion of questions answered correctly, adjudicated by
GPT-4o-mini. The judge sees the question, the gold answer, and the prediction,
and decides whether the prediction conveys the same information. Exact-match
scoring would penalise a correct answer for phrasing, which is why the paper
uses a judge and why this does too.

The judge is a separate provider instance so its tokens never contaminate the
memory-system token counts being compared.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sozograph.providers import LLMProvider

JUDGE_SYSTEM_PROMPT = """
You grade answers to questions about a long conversation.

You are given the question, the correct answer, and a candidate answer. Decide
whether the candidate conveys the same information as the correct answer.

Grade CORRECT when:
- The candidate states the same fact, even in different words.
- The candidate is more specific but consistent with the correct answer.
- The candidate contains the correct answer alongside extra detail that does
  not contradict it.
- Dates, names, or numbers match in substance ("May 8th" and "8 May").

Grade INCORRECT when:
- The candidate states something different or contradictory.
- The candidate says it does not know, while a real answer exists.
- The candidate is so vague that it does not actually answer the question.

Judge only whether the information matches. Style, length, and fluency are
irrelevant.
""".strip()

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}

JUDGE_USER_TEMPLATE = """
QUESTION:
{question}

CORRECT ANSWER:
{gold}

CANDIDATE ANSWER:
{prediction}

Does the candidate convey the same information as the correct answer?
""".strip()


@dataclass
class Verdict:
    correct: bool
    reason: str = ""


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def judge(
    provider: LLMProvider,
    *,
    question: str,
    gold: str,
    prediction: str,
) -> Verdict:
    """Grade one prediction."""
    if not str(prediction or "").strip():
        return Verdict(False, "empty prediction")

    # A trivially exact match needs no call. Saves judge spend on the easy ones
    # and cannot change the outcome.
    gold_norm, pred_norm = _normalize(gold), _normalize(prediction)
    if gold_norm and gold_norm == pred_norm:
        return Verdict(True, "exact match")

    payload = provider.complete_json(
        system=JUDGE_SYSTEM_PROMPT,
        user=JUDGE_USER_TEMPLATE.format(
            question=question, gold=gold, prediction=prediction
        ),
        schema=JUDGE_SCHEMA,
        temperature=0.0,
    )
    return Verdict(bool(payload.get("correct")), str(payload.get("reason") or ""))
