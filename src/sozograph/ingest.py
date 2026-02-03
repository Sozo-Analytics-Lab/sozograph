from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from google import genai
from google.genai import types

from .interaction import Interaction
from .schema import Passport, SourceRef
from .utils import utcnow, sha256_json, parse_ts, safe_stringify
from .adapters.firestore import firestore_to_interaction, firestore_batch_to_interactions
from .adapters.rtdb import rtdb_to_interaction, rtdb_batch_to_interactions
from .adapters.supabase import supabase_row_to_interaction, supabase_batch_to_interactions
from .prompts import (
    FALLBACK_SUMMARIZER_SYSTEM_PROMPT,
    FALLBACK_SUMMARIZER_USER_PROMPT_TEMPLATE,
)


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


def _looks_like_rtdb_envelope(obj: Dict[str, Any]) -> bool:
    return "path" in obj and ("value" in obj or "data" in obj)


def _looks_like_supabase_envelope(obj: Dict[str, Any]) -> bool:
    return "table" in obj and ("row" in obj or "data" in obj)


def _guess_hint(obj: Dict[str, Any]) -> str:
    """
    Best-effort hint detection when user doesn't specify.
    """
    if _looks_like_rtdb_envelope(obj):
        return "rtdb"
    if _looks_like_supabase_envelope(obj):
        return "supabase"
    # Firestore docs are just dicts; we treat default dicts as firestore-ish.
    return "firestore"


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


class FallbackSummarizer:
    """
    Gemini fallback summarizer used ONLY when we cannot derive meaningful text
    deterministically from an object.
    """

    def __init__(self, api_key: str, model: str = "gemini-3-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def summarize(
        self,
        obj: Any,
        *,
        source_hint: str,
        source_pointer: Optional[str],
        ts_iso: str,
    ) -> str:
        object_json = ""
        try:
            object_json = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
        except Exception:
            object_json = safe_stringify(obj)

        prompt = FALLBACK_SUMMARIZER_USER_PROMPT_TEMPLATE.format(
            source_hint=source_hint,
            source_pointer=source_pointer or "",
            ts_iso=ts_iso,
            object_json=object_json,
        )

        resp = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(role="system", parts=[types.Part(text=FALLBACK_SUMMARIZER_SYSTEM_PROMPT)]),
                types.Content(role="user", parts=[types.Part(text=prompt)]),
            ],
            config=types.GenerationContentConfig(
                temperature=0.2,
            ),
        )

        txt = (resp.text or "").strip()
        # Final guard: never return empty
        return txt if txt else "Database object (unstructured)."


def make_source_ref(
    *,
    source_id: str,
    kind: str,
    payload: Any,
    ts: Optional[Any] = None,
    source_pointer: Optional[str] = None,
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
    hint: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Interaction], List[SourceRef]]:
    """
    Convert arbitrary input into a list of Interactions + SourceRefs.

    This does NOT call the extractor. It only canonicalizes inputs.
    Gemini fallback summarization is applied later by apply_fallback_summaries().
    """
    meta = meta or {}
    interactions: List[Interaction] = []
    sources: List[SourceRef] = []

    # 1) String transcript
    if isinstance(item, str):
        src_id = meta.get("source_id") or f"t{abs(hash(item)) % 10_000_000}"
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

    # 3) Dict objects (DB docs / envelopes)
    if isinstance(item, dict):
        used_hint = (hint or item.get("_hint") or _guess_hint(item)).lower().strip()

        # RTDB envelope: {path, value}
        if used_hint == "rtdb" or _looks_like_rtdb_envelope(item):
            path = item.get("path") or meta.get("source") or meta.get("source_pointer")
            value = item.get("value", item.get("data"))
            it = rtdb_to_interaction(value, path=path)

            src_id = meta.get("source_id") or f"r{abs(hash(sha256_json(item))) % 10_000_000}"
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

            src_id = meta.get("source_id") or f"s{abs(hash(sha256_json(item))) % 10_000_000}"
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
                for it in its:
                    src_id = meta.get("source_id") or f"f{abs(hash(sha256_json(it.data))) % 10_000_000}"
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

            src_id = meta.get("source_id") or f"f{abs(hash(sha256_json(item))) % 10_000_000}"
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
        src_id = meta.get("source_id") or f"u{abs(hash(sha256_json(item))) % 10_000_000}"
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
    src_id = meta.get("source_id") or f"x{abs(hash(str(item))) % 10_000_000}"
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
    interactions: List[Interaction],
    *,
    sources: List[SourceRef],
    api_key: Optional[str],
    cfg: IngestConfig,
    fallback_model: str = "gemini-3-flash",
) -> List[Interaction]:
    """
    For any interaction whose text is too weak/noisy, optionally call Gemini fallback
    summarizer to get a better Interaction.text. This minimizes user pain.

    We DO NOT change Interaction.data; only improve Interaction.text.
    """
    if not cfg.enable_fallback_summarizer:
        return interactions
    if not api_key:
        return interactions

    summarizer = FallbackSummarizer(api_key=api_key, model=fallback_model)

    # Map source id by interaction id/source pointer best-effort
    # (In v1 we keep this simple: use first matching source if possible)
    src_by_pointer: Dict[str, SourceRef] = {}
    for s in sources:
        if s.source:
            src_by_pointer[s.source] = s

    out: List[Interaction] = []
    for it in interactions:
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


# ---------------------------------------------------------------------------
# ✅ v1 Public API: ingest()
# ---------------------------------------------------------------------------

def ingest(
    passport: Passport,
    item: Any,
    *,
    hint: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    cfg: Optional[IngestConfig] = None,
    fallback_model: str = "gemini-3-flash",
) -> Tuple[Passport, List[Interaction]]:
    """
    v1 ingestion entry-point.

    - Accepts: transcript string, list of transcripts, Firestore/RTDB/Supabase objects, or mixed list.
    - Canonicalizes input -> Interactions + SourceRefs
    - Optionally improves weak Interaction.text using Gemini fallback summarizer (NOT the extractor)
    - Always upserts sources into passport, and touches updated_at.

    Returns (passport, interactions) so the caller can pass interactions into the extractor step.
    """
    cfg = cfg or load_ingest_config()

    interactions, sources = coerce_to_interactions(item, hint=hint, meta=meta)

    # optional fallback text improvement
    interactions = apply_fallback_summaries(
        interactions,
        sources=sources,
        api_key=api_key,
        cfg=cfg,
        fallback_model=fallback_model,
    )

    # record sources on passport
    for s in sources:
        passport.upsert_source(s)

    passport.touch()
    return passport, interactions
