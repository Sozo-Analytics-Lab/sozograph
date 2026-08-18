from __future__ import annotations

from typing import Any

from .schema import ENTITY_TYPES

EXTRACTOR_SYSTEM_PROMPT = """
You are the SozoGraph extractor.

You convert interaction text into a compact, structured belief-state update.

Core philosophy:
- Extract beliefs, not quotes.
- Separate facts (what is true now) from preferences (what the person likes or wants).
- Track entities (people, projects, orgs, tools, places) and their aliases.
- Capture open loops (unresolved questions, pending tasks, missing information).
- Write one episode summarizing what happened, so a later question about
  this stretch of conversation can still be answered.
- When a value changes, emit the new value. The system resolves contradictions itself.

Rules:
- Reuse a key from KNOWN KEYS whenever the new information belongs to it. Inventing a
  synonym for a key that already exists fragments the memory and is the single most
  damaging thing you can do here.
- Keys are short, lowercase, snake_case.
- Confidence is 0 to 1. Use lower confidence when inferring rather than reading.
- Be conservative. Include only what is stable or actionable.
- Never invent detail that is not present in the text.
- Do not extract transient chatter, greetings, or one-off small talk as facts.
- The episode summary is different: it records what was discussed, including
  specifics such as names, places, numbers, and dates that would otherwise
  be lost. Write it so it stands alone without the original text.
""".strip()


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
        },
        # OpenAI strict mode requires every property listed in `required` and
        # additionalProperties false at every level.
        "required": ["key", "value", "confidence"],
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
        },
        "prefs": {
            "type": "array",
            "description": "Stable preferences: tone, style, language, constraints.",
            "items": _kv_item_schema(),
        },
        "entities": {
            "type": "array",
            "description": "Named entities worth remembering.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "type", "aliases"],
                "additionalProperties": False,
            },
        },
        "open_loops": {
            "type": "array",
            "description": "Unresolved questions or pending tasks.",
            "items": {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
                "additionalProperties": False,
            },
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
                "participants": {"type": "array", "items": {"type": "string"}},
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Distinctive terms someone might search for.",
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
    "required": ["facts", "prefs", "entities", "open_loops", "episode"],
    "additionalProperties": False,
}


EXTRACTOR_USER_PROMPT_TEMPLATE = """
{known_keys_block}
INTERACTION TYPE: {interaction_type}
TIMESTAMP: {ts_iso}

TEXT:
{interaction_text}

Extract the stable, useful updates from this text, and summarize what happened.
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
