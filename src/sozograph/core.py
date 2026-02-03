from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

from .schema import Passport
from .interaction import Interaction
from .ingest import (
    load_ingest_config,
    coerce_to_interactions,
    apply_fallback_summaries,
)
from .extractor import Extractor
from .resolver import merge_passport_update, ResolveStats
from .render import export_context as _export_context


def _require_api_key(passed: Optional[str]) -> str:
    api_key = passed or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Set it in environment or pass api_key=... to SozoGraph()."
        )
    return api_key


def _default_extractor_model() -> str:
    return os.getenv("SOZOGRAPH_EXTRACTOR_MODEL", "gemini-3-flash")


def _default_context_budget() -> int:
    try:
        return int(os.getenv("SOZOGRAPH_DEFAULT_CONTEXT_BUDGET", "3000"))
    except Exception:
        return 3000


class SozoGraph:
    """
    SozoGraph v1: transcript/db object -> portable cognitive passport JSON.

    Objects-only ingestion:
    - You fetch from Firestore/RTDB/Supabase however you want
    - You pass dict/list/string objects here
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        extractor_model: Optional[str] = None,
        fallback_model: str = "gemini-3-flash",
        enable_fallback_summarizer: Optional[bool] = None,
        max_interaction_chars: Optional[int] = None,
    ):
        self.api_key = _require_api_key(api_key)
        self.extractor_model = extractor_model or _default_extractor_model()
        self.fallback_model = fallback_model

        cfg = load_ingest_config()
        if enable_fallback_summarizer is not None:
            cfg.enable_fallback_summarizer = bool(enable_fallback_summarizer)
        if max_interaction_chars is not None:
            cfg.max_interaction_chars = int(max_interaction_chars)
        self.ingest_cfg = cfg

        self.extractor = Extractor(api_key=self.api_key, model=self.extractor_model)

    def ingest(
        self,
        data: Any,
        *,
        passport: Optional[Passport] = None,
        meta: Optional[Dict[str, Any]] = None,
        hint: Optional[str] = None,
    ) -> Tuple[Passport, List[ResolveStats]]:
        """
        Ingest any supported input and return updated Passport + per-interaction stats.

        data can be:
        - str (transcript)
        - dict (firestore doc / rtdb snapshot envelope / supabase row envelope)
        - list (mixed)
        """
        base = passport or Passport()
        meta = meta or {}

        interactions, sources = coerce_to_interactions(data, hint=hint, meta=meta)

        # Attach sources to passport (unique by id)
        for s in sources:
            base.upsert_source(s)

        # Improve weak texts via Gemini fallback summarizer (optional)
        interactions = apply_fallback_summaries(
            interactions,
            sources=sources,
            api_key=self.api_key,
            cfg=self.ingest_cfg,
            fallback_model=self.fallback_model,
        )

        # Extract + merge sequentially (temporal truth)
        stats_list: List[ResolveStats] = []
        for idx, it in enumerate(interactions):
            # Pick the closest source id for this interaction.
            # In v1 we keep it deterministic: use meta.source_id if provided, else stable index-based.
            source_id = meta.get("source_id")
            if not source_id:
                # Try find a matching SourceRef by pointer; otherwise create a stable one
                if it.source:
                    # If user provided source pointer, use it to derive stable id
                    source_id = f"src_{abs(hash(it.source)) % 10_000_000}"
                else:
                    source_id = f"i_{idx}"

            update = self.extractor.extract(it, source_id=source_id)
            base, stats = merge_passport_update(
                base,
                facts=update["facts"],
                prefs=update["prefs"],
                entities=update["entities"],
                open_loops=update["open_loops"],
            )
            stats_list.append(stats)

        return base, stats_list

    def export_context(
        self,
        passport: Passport,
        *,
        budget_chars: Optional[int] = None,
        header: str = "SOZOGRAPH PASSPORT v1",
    ) -> str:
        """
        Export a compact briefing string suitable for agent context injection.
        """
        return _export_context(
            passport,
            budget_chars=budget_chars or _default_context_budget(),
            header=header,
        )
