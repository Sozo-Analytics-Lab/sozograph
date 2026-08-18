from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .adapters.firestore import firestore_batch_to_interactions, firestore_to_interaction
from .adapters.rtdb import rtdb_to_interaction
from .adapters.supabase import supabase_row_to_interaction
from .interaction import Interaction
from .prompts import SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_USER_PROMPT_TEMPLATE
from .providers.base import LLMProvider
from .schema import SourceRef
from .utils import parse_ts, pick_first, safe_stringify, sha256_json, stable_id, utcnow


@dataclass
class IngestConfig:
    enable_fallback_summarizer: bool = True
    max_interaction_chars: int = 4000


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def load_ingest_config() -> IngestConfig:
    return IngestConfig(
        enable_fallback_summarizer=_env_bool("SOZOGRAPH_ENABLE_FALLBACK_SUMMARIZER", True),
        max_interaction_chars=int(os.getenv("SOZOGRAPH_MAX_INTERACTION_CHARS", "4000")),
    )


def _looks_like_chat_turn(obj: dict[str, Any]) -> bool:
    """
    A conversational turn: some text plus, usually, who said it.

    Conversation is the primary thing this library remembers, so the shape
    deserves first-class handling rather than falling through to a generic
    key-value stringify that buries the utterance in field names.
    """
    has_text = any(isinstance(obj.get(k), str) and obj[k].strip()
                   for k in ("text", "content", "message", "utterance"))
    if not has_text:
        return False
    if any(k in obj for k in ("speaker", "role", "author", "from", "user")):
        return True
    # A bare {text, timestamp} pair is still a turn.
    return len(set(obj) - {"text", "content", "message", "utterance",
                           "ts", "timestamp", "time", "date", "id",
                           "session", "session_id", "dia_id"}) == 0


def _looks_like_rtdb_envelope(obj: dict[str, Any]) -> bool:
    return "path" in obj and ("value" in obj or "data" in obj)


def _looks_like_supabase_envelope(obj: dict[str, Any]) -> bool:
    return "table" in obj and ("row" in obj or "data" in obj)


def _guess_hint(obj: dict[str, Any]) -> str:
    """
    Best-effort hint detection when user doesn't specify.
    """
    if _looks_like_rtdb_envelope(obj):
        return "rtdb"
    if _looks_like_supabase_envelope(obj):
        return "supabase"
    # Firestore docs are just dicts; we treat default dicts as firestore-ish.
    return "firestore"


#: Interaction types whose text came from a person and is already readable.
_HUMAN_TEXT_TYPES = frozenset({"chat", "transcript", "note", "email", "message"})


def _is_text_too_weak(text: str) -> bool:
    """
    Decide whether deterministic text is too weak and needs Gemini fallback.
    We keep this simple and conservative in v1.
    """
    if not text:
        return True
    t = text.strip()
    if len(t) < 30:
        return True
    # If it looks like "key: val; key: val" only, we may still accept it;
    # but if it's mostly punctuation/noise, fallback.
    alnum = sum(ch.isalnum() for ch in t)
    if alnum / max(len(t), 1) < 0.35:
        return True
    return False


class Summarizer:
    """
    Turns an unreadable database object into text worth extracting from.

    Used only when the deterministic path cannot produce meaningful text, so
    most ingestions never call a model here at all.
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def summarize(
        self,
        obj: Any,
        *,
        source_hint: str,
        source_pointer: str | None,
        ts_iso: str,
    ) -> str:
        try:
            object_json = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            object_json = safe_stringify(obj)

        text = self.provider.complete_text(
            system=SUMMARIZER_SYSTEM_PROMPT,
            user=SUMMARIZER_USER_PROMPT_TEMPLATE.format(
                source_hint=source_hint,
                source_pointer=source_pointer or "",
                ts_iso=ts_iso,
                object_json=object_json,
            ),
            temperature=0.2,
        ).strip()
        return text or "Database object (unstructured)."


# Kept so existing imports keep resolving for one release.
FallbackSummarizer = Summarizer


def make_source_ref(
    *,
    source_id: str,
    kind: str,
    payload: Any,
    ts: Any | None = None,
    source_pointer: str | None = None,
) -> SourceRef:
    dt = parse_ts(ts) or utcnow()
    return SourceRef(
        id=source_id,
        kind=kind,  # validated later by pydantic in Passport
        ts=dt,
        hash=sha256_json(payload),
        source=source_pointer,
    )


def coerce_to_interactions(
    item: Any,
    *,
    hint: str | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[list[Interaction], list[SourceRef]]:
    """
    Convert arbitrary input into a list of Interactions + SourceRefs.

    This does NOT call the extractor. It only canonicalizes inputs.
    Gemini fallback summarization is applied later by apply_fallback_summaries().
    """
    meta = meta or {}
    interactions: list[Interaction] = []
    sources: list[SourceRef] = []

    # 1) String transcript
    if isinstance(item, str):
        src_id = meta.get("source_id") or stable_id("t", item)
        src_ptr = meta.get("source") or meta.get("source_pointer")
        ts = parse_ts(meta.get("ts")) or utcnow()

        interactions.append(
            Interaction(
                id=meta.get("id"),
                ts=ts,
                type=meta.get("type", "transcript"),
                text=item,
                source=src_ptr,
                data=None,
                meta=meta,
            )
        )
        sources.append(
            make_source_ref(
                source_id=src_id,
                kind=meta.get("kind", "transcript"),
                payload={"text": item, "meta": meta},
                ts=ts,
                source_pointer=src_ptr,
            )
        )
        return interactions, sources

    # 2) List of mixed items
    if isinstance(item, list):
        for idx, sub in enumerate(item):
            sub_meta = dict(meta)
            # allow per-item override without forcing shape
            sub_meta.setdefault("source_id", f"{meta.get('source_id','h')}_{idx}")
            sub_interactions, sub_sources = coerce_to_interactions(sub, hint=hint, meta=sub_meta)
            interactions.extend(sub_interactions)
            sources.extend(sub_sources)
        return interactions, sources

    # 3) Dict objects (DB docs / envelopes / chat turns)
    if isinstance(item, dict):
        used_hint = (hint or item.get("_hint") or _guess_hint(item)).lower().strip()

        if used_hint == "chat" or (hint is None and _looks_like_chat_turn(item)):
            text = pick_first(item, ("text", "content", "message", "utterance")) or ""
            speaker = pick_first(item, ("speaker", "role", "author", "from", "user"))
            ts = (parse_ts(pick_first(item, ("ts", "timestamp", "time", "date")))
                  or parse_ts(meta.get("ts")) or utcnow())
            turn_meta = dict(meta)
            if speaker:
                turn_meta["speaker"] = str(speaker)
            session = pick_first(item, ("session", "session_id", "thread_id"))
            if session is not None:
                turn_meta["session"] = str(session)

            src_id = meta.get("source_id") or stable_id("c", item)
            src_ptr = meta.get("source") or meta.get("source_pointer")
            interactions.append(
                Interaction(
                    id=str(item.get("id") or item.get("dia_id") or sha256_json(item)[:16]),
                    ts=ts,
                    type="chat",
                    text=str(text),
                    source=src_ptr,
                    data=item,
                    meta=turn_meta,
                )
            )
            sources.append(
                make_source_ref(
                    source_id=src_id, kind="chat", payload=item,
                    ts=ts, source_pointer=src_ptr,
                )
            )
            return interactions, sources

        # RTDB envelope: {path, value}
        if used_hint == "rtdb" or _looks_like_rtdb_envelope(item):
            path = item.get("path") or meta.get("source") or meta.get("source_pointer")
            value = item.get("value", item.get("data"))
            it = rtdb_to_interaction(value, path=path)

            src_id = meta.get("source_id") or stable_id("r", item)
            sources.append(
                make_source_ref(
                    source_id=src_id,
                    kind="rtdb",
                    payload=item,
                    ts=it.ts,
                    source_pointer=it.source,
                )
            )
            interactions.append(it)
            return interactions, sources

        # Supabase envelope: {table, row}
        if used_hint == "supabase" or _looks_like_supabase_envelope(item):
            table = item.get("table") or meta.get("table")
            row = item.get("row", item.get("data", item))
            it = supabase_row_to_interaction(row if isinstance(row, dict) else {"value": row}, table=table)

            src_id = meta.get("source_id") or stable_id("s", item)
            sources.append(
                make_source_ref(
                    source_id=src_id,
                    kind="supabase",
                    payload=item,
                    ts=it.ts,
                    source_pointer=it.source,
                )
            )
            interactions.append(it)
            return interactions, sources

        # Firestore: doc dict OR batch dict/list
        if used_hint == "firestore":
            # batch dict mapping {doc_id: doc}
            if all(isinstance(v, dict) for v in item.values()) and any(k for k in item.keys()):
                # ambiguous: could be a single doc with many nested dicts; we treat as batch
                col_path = meta.get("source") or meta.get("collection_path")
                its = firestore_batch_to_interactions(item, collection_path=col_path)
                # One source per interaction for traceability
                base_src_id = meta.get("source_id")
                for doc_idx, it in enumerate(its):
                    # Suffix a caller-supplied id: without it every doc in the
                    # batch shares one SourceRef id and upsert_source keeps only
                    # the last.
                    src_id = (
                        f"{base_src_id}_{doc_idx}" if base_src_id
                        else stable_id("f", it.data)
                    )
                    sources.append(
                        make_source_ref(
                            source_id=src_id,
                            kind="firestore",
                            payload=it.data,
                            ts=it.ts,
                            source_pointer=it.source,
                        )
                    )
                    interactions.append(it)
                return interactions, sources

            # single doc
            doc_id = item.get("id") or meta.get("id")
            src_ptr = meta.get("source") or meta.get("source_pointer") or None
            it = firestore_to_interaction(item, source=src_ptr, doc_id=doc_id)

            src_id = meta.get("source_id") or stable_id("f", item)
            sources.append(
                make_source_ref(
                    source_id=src_id,
                    kind="firestore",
                    payload=item,
                    ts=it.ts,
                    source_pointer=it.source,
                )
            )
            interactions.append(it)
            return interactions, sources

        # Unknown dict: treat as generic event
        text = safe_stringify(item)
        ts = parse_ts(item.get("ts") if isinstance(item, dict) else None) or utcnow()
        src_id = meta.get("source_id") or stable_id("u", item)
        src_ptr = meta.get("source") or meta.get("source_pointer")

        interactions.append(
            Interaction(
                id=meta.get("id") or item.get("id") or sha256_json(item)[:16],
                ts=ts,
                type=meta.get("type", "unknown"),
                text=text,
                source=src_ptr,
                data=item,
                meta=meta,
            )
        )
        sources.append(
            make_source_ref(
                source_id=src_id,
                kind=meta.get("kind", "unknown"),
                payload=item,
                ts=ts,
                source_pointer=src_ptr,
            )
        )
        return interactions, sources

    # 4) Fallback for other types
    text = safe_stringify(item)
    ts = parse_ts(meta.get("ts")) or utcnow()
    src_id = meta.get("source_id") or stable_id("x", str(item))
    src_ptr = meta.get("source") or meta.get("source_pointer")
    interactions.append(
        Interaction(
            id=meta.get("id") or sha256_json({"v": str(item)})[:16],
            ts=ts,
            type=meta.get("type", "unknown"),
            text=text,
            source=src_ptr,
            data={"value": str(item)},
            meta=meta,
        )
    )
    sources.append(
        make_source_ref(
            source_id=src_id,
            kind=meta.get("kind", "unknown"),
            payload={"value": str(item), "meta": meta},
            ts=ts,
            source_pointer=src_ptr,
        )
    )
    return interactions, sources


def apply_fallback_summaries(
    interactions: list[Interaction],
    *,
    sources: list[SourceRef],
    provider: LLMProvider | None,
    cfg: IngestConfig,
) -> list[Interaction]:
    """
    Improve any interaction whose text is too weak to extract from.

    Only Interaction.text changes; Interaction.data is left untouched so the
    evidence hash still refers to the original payload.
    """
    if not cfg.enable_fallback_summarizer or provider is None:
        return interactions

    summarizer = Summarizer(provider)

    # Map source id by interaction id/source pointer best-effort
    # (In v1 we keep this simple: use first matching source if possible)
    src_by_pointer: dict[str, SourceRef] = {}
    for s in sources:
        if s.source:
            src_by_pointer[s.source] = s

    out: list[Interaction] = []
    for it in interactions:
        # Human text is never summarized. "What colour did you go with?" is 28
        # characters and perfectly clear; running it through a model to be told
        # so costs a call per short turn and changes nothing. The summarizer
        # exists for database objects that stringify into noise.
        if it.type in _HUMAN_TEXT_TYPES:
            out.append(it)
            continue

        txt = it.text or ""
        # truncate before evaluating (avoid massive stringify)
        if len(txt) > cfg.max_interaction_chars:
            txt = txt[: cfg.max_interaction_chars - 1] + "…"
            it.text = txt

        if not _is_text_too_weak(it.text):
            out.append(it)
            continue

        # Summarize the raw object if present, else summarize the weak text
        payload = it.data if it.data is not None else {"text": it.text}

        improved = summarizer.summarize(
            payload,
            source_hint=it.type,
            source_pointer=it.source,
            ts_iso=it.ts.isoformat(),
        )

        it.text = improved[: cfg.max_interaction_chars] if improved else it.text
        out.append(it)

    return out
