from __future__ import annotations

from typing import Any

from .schema import ENTITY_TYPES

EXTRACTOR_SYSTEM_PROMPT = """
You are the SozoGraph extractor.

You convert interaction text into a compact, structured memory update with two
layers: a small belief state and a set of atomic observations.

Core philosophy:
- Facts are the belief state: stable truths, one per key, that later overwrite
  each other when they change (role, location, tools owned, project status).
- Preferences are what the person likes or wants (tone, style, constraints).
- Observations are the recall layer: atomic, concrete details of what was said
  or happened that a fact would throw away but a later question may ask for
  ("Oliver hid his bone in Melanie's slipper", "the trail hike took two hours",
  "Tim is learning German"). Capture these generously. They are the difference
  between remembering that someone exists and remembering what they told you.
- Track entities (people, projects, orgs, tools, places) and their aliases.
- Capture open loops (unresolved questions, pending tasks, missing information).
- Write one episode summarizing what happened across this stretch.
- When a fact's value changes, emit the new value. The system resolves
  contradictions itself.

Rules for facts and preferences:
- Reuse a key from KNOWN KEYS whenever the new information belongs to it. Inventing a
  synonym for a key that already exists fragments the memory and is the single most
  damaging thing you can do here.
- Keys are short, lowercase, snake_case.
- Confidence is 0 to 1. Use lower confidence when inferring rather than reading.
- Be conservative here. A fact is something that stays true; put one-off detail
  in observations instead, not in facts.

Rules for observations:
- Each observation is ONE self-contained statement, written in the third person,
  understandable with no other context. Resolve every pronoun to a named person
  ("she" -> "Melanie") before writing it.
- Prefer many small observations over one dense sentence. Split "they hiked for
  two hours and saw a rainbow" into two observations.
- Cover the concrete specifics: events, places, quantities, choices, quotes,
  plans, one-off details. Skip only pure greetings and filler with no content.
- Do not restate a fact or preference you already emitted; observations are for
  what the belief state cannot hold.

Rules for everything:
- Never invent detail that is not present in the text.
- For every extracted item, copy the shortest verbatim source substring that
  supports it into evidence_quote. Preserve spelling and punctuation exactly.
  Use an empty string only when no single substring supports a derived item.
- The TIMESTAMP above is "now". Resolve every relative date or duration in the
  text ("yesterday", "last Saturday", "in two weeks") into an absolute calendar
  date computed from that timestamp before writing it anywhere, in observations
  and the episode summary alike.
""".strip()


#: Hard bounds on the extraction arrays.
#:
#: Grammar-constrained decoding removes the model's own "I'm done" signal at
#: each array element: it only ever chooses "continue the array or close it".
#: A small model can pick "continue" indefinitely, restating near-duplicate
#: items until the completion cap kills the call mid-JSON. Four conversations
#: crashed this way in one eval run. A schema-level bound terminates the loop
#: by construction, and every supported structured-output dialect (OpenAI
#: strict mode, Gemini response_schema, Ollama format) accepts maxItems.
ARRAY_LIMITS: dict[str, int] = {
    "facts": 24,
    "prefs": 16,
    "entities": 12,
    "open_loops": 10,
    "observations": 30,
    "aliases": 6,
    "participants": 8,
    "keywords": 10,
}


def _kv_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Short snake_case identifier. Reuse a known key when one fits.",
            },
            "value": {
                "type": "string",
                "description": "The value. Numbers and booleans are written as text.",
            },
            "confidence": {
                "type": "number",
                "description": "0 to 1. Lower when inferred rather than stated.",
            },
            "evidence_quote": {
                "type": "string",
                "description": "Shortest exact substring from TEXT that supports this item.",
            },
        },
        # OpenAI strict mode requires every property listed in `required` and
        # additionalProperties false at every level.
        "required": ["key", "value", "confidence", "evidence_quote"],
        "additionalProperties": False,
    }


#: The wire contract, as a real JSON Schema.
#:
#: This goes to each engine's native structured-output mechanism rather than
#: being pasted into the prompt as prose. Two deliberate shapes: `value` is a
#: string so the schema stays valid under OpenAI strict mode (the validator
#: coerces numbers and booleans back), and there is no `ts` or `source` field
#: because the system already knows both from the interaction. Asking the model
#: for a timestamp it cannot know was the source of a silent data-loss bug.
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "description": "Stable truths: role, location, tools owned, project status.",
            "items": _kv_item_schema(),
            "maxItems": ARRAY_LIMITS["facts"],
        },
        "prefs": {
            "type": "array",
            "description": "Stable preferences: tone, style, language, constraints.",
            "items": _kv_item_schema(),
            "maxItems": ARRAY_LIMITS["prefs"],
        },
        "entities": {
            "type": "array",
            "description": "Named entities worth remembering.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": ARRAY_LIMITS["aliases"],
                    },
                    "evidence_quote": {
                        "type": "string",
                        "description": "Shortest exact substring from TEXT naming this entity.",
                    },
                },
                "required": ["name", "type", "aliases", "evidence_quote"],
                "additionalProperties": False,
            },
            "maxItems": ARRAY_LIMITS["entities"],
        },
        "open_loops": {
            "type": "array",
            "description": "Unresolved questions or pending tasks.",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "evidence_quote": {
                        "type": "string",
                        "description": "Shortest exact substring from TEXT supporting the loop.",
                    },
                },
                "required": ["item", "evidence_quote"],
                "additionalProperties": False,
            },
            "maxItems": ARRAY_LIMITS["open_loops"],
        },
        "observations": {
            "type": "array",
            "description": (
                "Atomic, self-contained details of what was said or happened, "
                "one statement each, third person, pronouns and relative dates "
                "resolved. The recall layer a later question reads from."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "One self-contained statement, understandable alone.",
                    },
                    "when": {
                        "type": "string",
                        "description": (
                            "ISO date (YYYY-MM-DD) of when this happened, resolved "
                            "from TIMESTAMP and the text. Empty string if unclear."
                        ),
                    },
                    "evidence_quote": {
                        "type": "string",
                        "description": "Shortest exact substring from TEXT supporting the detail.",
                    },
                },
                "required": ["text", "when", "evidence_quote"],
                "additionalProperties": False,
            },
            "maxItems": ARRAY_LIMITS["observations"],
        },
        "episode": {
            "type": "object",
            "description": (
                "What happened in this stretch of conversation. One to three "
                "sentences, concrete, naming the specifics a person would need "
                "to answer a question about it later."
            ),
            "properties": {
                "summary": {"type": "string"},
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": ARRAY_LIMITS["participants"],
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Distinctive terms someone might search for.",
                    "maxItems": ARRAY_LIMITS["keywords"],
                },
                "salience": {
                    "type": "number",
                    "description": "0 to 1. How much this matters later.",
                },
            },
            "required": ["summary", "participants", "keywords", "salience"],
            "additionalProperties": False,
        },
    },
    "required": ["facts", "prefs", "entities", "open_loops", "observations", "episode"],
    "additionalProperties": False,
}


EXTRACTOR_USER_PROMPT_TEMPLATE = """
{known_keys_block}
INTERACTION TYPE: {interaction_type}
TIMESTAMP: {ts_iso}

TEXT:
{interaction_text}

Extract the belief-state updates, the atomic observations, and a summary of
what happened. Be generous with observations: they are what a later question reads.
""".strip()


def format_known_keys(keys: list[str], *, limit: int = 120) -> str:
    """
    Render the passport's existing keys for the prompt.

    This is the cheapest deduplication mechanism in the system and it costs no
    extra API call. An extractor that can see `code_style` already exists will
    reuse it rather than inventing `boilerplate_preference`; string distance
    cannot catch that pair, and an LLM reconciliation pass to merge it later
    costs a round trip. Preventing the split is better than repairing it.
    """
    if not keys:
        return ""
    shown = keys[:limit]
    joined = ", ".join(shown)
    more = f" (+{len(keys) - len(shown)} more)" if len(keys) > len(shown) else ""
    return (
        "KNOWN KEYS (reuse these exact keys when the new information belongs to one "
        f"of them):\n{joined}{more}\n"
    )


SUMMARIZER_SYSTEM_PROMPT = """
You are the SozoGraph summarizer.

You are given an arbitrary object from a database. Write a compact, human-readable
summary that captures its meaning without dumping raw blobs or opaque identifiers.

Rules:
- Plain text only. No JSON, no markdown.
- Two to eight lines.
- Focus on who, what, when, status, decision, outcome.
- Skip internal identifiers unless they are meaningful to a person.
- If the object is mostly noise, say what it represents at a high level.
""".strip()

SUMMARIZER_USER_PROMPT_TEMPLATE = """
SOURCE: {source_hint}
POINTER: {source_pointer}
TIMESTAMP: {ts_iso}

OBJECT:
{object_json}

Write a compact summary suitable for a memory system.
""".strip()
