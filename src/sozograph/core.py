from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from .batching import DEFAULT_MAX_TOKENS, segment_interactions
from .batching import plan as plan_batches
from .extractor import Extractor
from .ingest import apply_fallback_summaries, coerce_to_interactions, load_ingest_config
from .passport3 import (
    Authority,
    MemoryPassport,
    ProjectionStats,
    Sensitivity,
    project_extraction_update,
)
from .providers import LLMProvider, from_env, get_provider
from .render import export_context as _export_context
from .resolver import ResolveStats, merge_passport_update
from .schema import Passport, SourceRef
from .utils import sha256_json, stable_id


def _default_context_budget() -> int:
    try:
        return int(os.getenv("SOZOGRAPH_DEFAULT_CONTEXT_BUDGET", "6000"))
    except ValueError:
        return 6000


class SozoGraph:
    """
    Compress conversation history into a portable JSON passport.

        sg = SozoGraph()                       # provider from the environment
        passport = sg.ingest(transcript)
        print(passport.context())
        passport.save("user.json")

    The provider is built on first use, so constructing a SozoGraph never needs
    a key and never touches the network. Loading, querying, and saving an
    existing passport work with no SDK installed at all.
    """

    def __init__(
        self,
        provider: Any | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        enable_fallback_summarizer: bool | None = None,
        max_interaction_chars: int | None = None,
        **provider_kwargs: Any,
    ):
        """
        `provider` accepts a spec string ("openai", "anthropic:claude-opus-5"),
        an already-built LLMProvider, or None to resolve from the environment.
        """
        self._provider: LLMProvider | None = None
        self._provider_spec = provider
        self._provider_kwargs = dict(provider_kwargs)
        if model is not None:
            self._provider_kwargs["model"] = model
        if api_key is not None:
            self._provider_kwargs["api_key"] = api_key

        if isinstance(provider, LLMProvider):
            self._provider = provider

        cfg = load_ingest_config()
        if enable_fallback_summarizer is not None:
            cfg.enable_fallback_summarizer = bool(enable_fallback_summarizer)
        if max_interaction_chars is not None:
            cfg.max_interaction_chars = int(max_interaction_chars)
        self.ingest_cfg = cfg

    # -- provider ---------------------------------------------------------

    @property
    def provider(self) -> LLMProvider:
        """The LLM provider, built on first access."""
        if self._provider is None:
            if isinstance(self._provider_spec, str):
                self._provider = get_provider(self._provider_spec, **self._provider_kwargs)
            else:
                self._provider = from_env(**self._provider_kwargs)
        return self._provider

    @property
    def usage(self):
        """Cumulative token and call counts, or None if nothing ran yet."""
        return self._provider.usage if self._provider is not None else None

    # -- ingestion --------------------------------------------------------

    def ingest(
        self,
        data: Any,
        *,
        passport: Passport | None = None,
        meta: dict[str, Any] | None = None,
        hint: str | None = None,
        batch: bool = True,
        max_segment_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Passport:
        """
        Ingest a transcript, a database object, or a list of either.

        Returns the updated Passport. Merge statistics are on `passport.stats`.

        Interactions are batched into token-bounded segments by default, one
        extraction call each. Pass `batch=False` to extract per interaction,
        which costs one call per turn and is almost never what you want.
        """
        base = passport if passport is not None else Passport.new()
        meta = meta or {}

        user_key = meta.get("user_key")
        if user_key:
            base.user_key = str(user_key)

        interactions, sources = coerce_to_interactions(data, hint=hint, meta=meta)
        extractor = Extractor(self.provider)
        interactions = apply_fallback_summaries(
            interactions,
            sources=sources,
            provider=self.provider,
            cfg=self.ingest_cfg,
        )

        stats_list: list[ResolveStats] = []

        if batch:
            units = segment_interactions(interactions, max_tokens=max_segment_tokens)
            # Provenance is recorded per segment, matching the granularity the
            # facts actually cite. One SourceRef per turn made the evidence log
            # larger than the memory it documented on a long conversation, and
            # nothing referenced those entries.
            for segment in units:
                base.upsert_source(
                    SourceRef(
                        id=stable_id("seg_", segment.id),
                        kind=_source_kind(segment.type),
                        ts=segment.ts,
                        hash=sha256_json([i.text for i in segment.interactions]),
                        source=segment.source,
                    )
                )
            for segment in units:
                # Tier 0 deduplication: show the model the vocabulary it already
                # has so it reuses a key rather than coining a synonym. Read
                # fresh each round so keys learned a moment ago are visible.
                #
                # One segment's extraction can fail on its own: a provider
                # timeout under sustained load, or a completion the engine
                # truncated mid-JSON on an unusually dense stretch. Losing the
                # whole conversation's memory because segment 30 of 34 hiccuped
                # is the wrong failure. Skip the segment, record it, and keep
                # the passport built so far -- the same "keep what finished"
                # contract the benchmark runner already honours per conversation.
                try:
                    update = extractor.extract_segment(
                        segment, known_keys=base.known_keys()
                    )
                except Exception as exc:  # noqa: BLE001 - provider/decoding failure is opaque here
                    _record_segment_failure(base, segment, exc)
                    continue
                base, stats = merge_passport_update(base, **_update_kwargs(update))
                stats_list.append(stats)
        else:
            for src in sources:
                base.upsert_source(src)
            for idx, it in enumerate(interactions):
                source_id = meta.get("source_id")
                if not source_id:
                    source_id = stable_id("src_", it.source) if it.source else f"i_{idx}"
                elif len(interactions) > 1:
                    source_id = f"{source_id}_{idx}"

                update = extractor.extract(
                    it, source_id=source_id, known_keys=base.known_keys()
                )
                base, stats = merge_passport_update(base, **_update_kwargs(update))
                stats_list.append(stats)

        base.stats = stats_list
        return base

    def ingest_v3(
        self,
        data: Any,
        *,
        passport: MemoryPassport | None = None,
        meta: dict[str, Any] | None = None,
        hint: str | None = None,
        batch: bool = True,
        max_segment_tokens: int = DEFAULT_MAX_TOKENS,
        replica_id: str | None = None,
        passport_id: str | None = None,
        transaction_time: datetime | None = None,
        authority: Authority = "inferred",
        sensitivity: Sensitivity = "internal",
        scopes: list[str] | None = None,
    ) -> MemoryPassport:
        """Ingest directly into the Passport 3 event ledger.

        The extractor emits validated candidates. The projector immediately
        writes immutable events with source evidence, valid time, transaction
        time, policy fields, and stable logical identities.
        """
        meta = meta or {}
        interactions, sources = coerce_to_interactions(data, hint=hint, meta=meta)
        interactions = apply_fallback_summaries(
            interactions,
            sources=sources,
            provider=self.provider,
            cfg=self.ingest_cfg,
        )
        stamp = transaction_time or datetime.now(timezone.utc)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        subject_key = str(meta["user_key"]) if meta.get("user_key") else None
        if passport is None:
            base = MemoryPassport.new(
                subject_key=subject_key,
                replica_id=replica_id or str(meta.get("replica_id") or "local"),
                passport_id=passport_id,
                now=stamp,
            )
        else:
            base = passport
            if subject_key and base.subject_key and subject_key != base.subject_key:
                raise ValueError("meta user_key does not match the Passport 3 subject_key")
            if subject_key and not base.subject_key:
                base.subject_key = subject_key
            if passport_id and passport_id != base.passport_id:
                raise ValueError("passport_id does not match the supplied Passport 3 ledger")
            if replica_id:
                base.replica_id = replica_id
                base.signatures = []

        extractor = Extractor(self.provider)
        totals = ProjectionStats()

        def known_keys() -> list[str]:
            return sorted({
                record.key
                for record in base.materialize()
                if record.kind in {"fact", "preference"} and record.key
            })

        def project(
            update: dict[str, Any],
            *,
            source_id: str,
            source_text: str,
            source_time: datetime,
            source_end_time: datetime | None,
            source_kind: str,
            source_pointer: str | None,
        ) -> None:
            stats = project_extraction_update(
                base,
                update,
                source_id=source_id,
                source_text=source_text,
                source_time=source_time,
                source_end_time=source_end_time,
                transaction_time=stamp,
                source_kind=source_kind,
                source_pointer=source_pointer,
                authority=authority,
                sensitivity=sensitivity,
                scopes=scopes,
            )
            totals.events_appended += stats.events_appended
            totals.revisions_skipped += stats.revisions_skipped
            totals.exact_evidence += stats.exact_evidence
            totals.coarse_evidence += stats.coarse_evidence

        if batch:
            for segment in segment_interactions(
                interactions,
                max_tokens=max_segment_tokens,
            ):
                source_text = segment.text(max_chars=12_000)
                source_id = stable_id(
                    "seg_",
                    {
                        "segment_id": segment.id,
                        "start": segment.ts.isoformat(),
                        "end": segment.end_ts.isoformat(),
                        "source": segment.source,
                        "text": source_text,
                    },
                )
                try:
                    update = extractor.extract_segment(
                        segment,
                        known_keys=known_keys(),
                    )
                except Exception as exc:  # noqa: BLE001
                    _record_v3_failure(base, segment.id, segment.ts, exc)
                    continue
                project(
                    update,
                    source_id=source_id,
                    source_text=source_text,
                    source_time=segment.ts,
                    source_end_time=segment.end_ts,
                    source_kind=_source_kind(segment.type),
                    source_pointer=segment.source,
                )
        else:
            for index, interaction in enumerate(interactions):
                source_id = meta.get("source_id")
                if not source_id:
                    source_id = stable_id(
                        "src_",
                        {
                            "id": interaction.id,
                            "source": interaction.source,
                            "timestamp": interaction.ts.isoformat(),
                            "text": interaction.short_text(),
                        },
                    )
                elif len(interactions) > 1:
                    source_id = f"{source_id}_{index}"
                source_text = interaction.short_text()
                try:
                    update = extractor.extract(
                        interaction,
                        source_id=str(source_id),
                        known_keys=known_keys(),
                    )
                except Exception as exc:  # noqa: BLE001
                    _record_v3_failure(base, str(source_id), interaction.ts, exc)
                    continue
                project(
                    update,
                    source_id=str(source_id),
                    source_text=source_text,
                    source_time=interaction.ts,
                    source_end_time=interaction.ts,
                    source_kind=_source_kind(interaction.type),
                    source_pointer=interaction.source,
                )

        base.extensions["sozograph:last_ingest"] = {
            **totals.to_dict(),
            "transaction_time": stamp.isoformat(),
        }
        base.signatures = []
        return base

    def plan(
        self,
        data: Any,
        *,
        meta: dict[str, Any] | None = None,
        hint: str | None = None,
        max_segment_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, float]:
        """
        Report what ingesting `data` will cost, without calling a model.

        Worth running once before ingesting a long history.
        """
        interactions, _ = coerce_to_interactions(data, hint=hint, meta=meta or {})
        return plan_batches(interactions, max_tokens=max_segment_tokens)

    # -- export -----------------------------------------------------------

    def export_context(
        self,
        passport: Passport,
        *,
        query: str | None = None,
        budget_chars: int | None = None,
        header: str = "SOZOGRAPH PASSPORT",
    ) -> str:
        """Render the passport as a context block for a prompt."""
        return _export_context(
            passport,
            query=query,
            budget_chars=budget_chars or _default_context_budget(),
            header=header,
        )


_SOURCE_KINDS = frozenset(
    {"transcript", "firestore", "rtdb", "supabase", "chat", "form", "unknown"}
)


def _source_kind(interaction_type: str) -> str:
    return interaction_type if interaction_type in _SOURCE_KINDS else "unknown"


def _record_segment_failure(base: Passport, segment: Any, exc: Exception) -> None:
    """
    Note a skipped segment on the passport so the loss is auditable, not silent.

    A dropped segment is real data loss; it belongs in the record next to the
    dedupe audit, where a caller inspecting the passport can see it happened and
    why, rather than discovering a hole by its absence.
    """
    failures = base.meta.setdefault("ingest_failures", [])
    failures.append({
        "segment": getattr(segment, "id", None),
        "ts": getattr(getattr(segment, "ts", None), "isoformat", lambda: None)(),
        "error": f"{type(exc).__name__}: {exc}"[:300],
    })


def _record_v3_failure(
    base: MemoryPassport,
    source_id: str,
    stamp: datetime,
    exc: Exception,
) -> None:
    failures = base.extensions.setdefault("sozograph:ingest_failures", [])
    failures.append({
        "source_id": source_id,
        "source_time": stamp.isoformat(),
        "error": f"{type(exc).__name__}: {exc}"[:300],
    })
    base.signatures = []


def _update_kwargs(update: dict[str, Any]) -> dict[str, Any]:
    return {
        "facts": update.get("facts") or [],
        "prefs": update.get("prefs") or [],
        "entities": update.get("entities") or [],
        "open_loops": update.get("open_loops") or [],
        "episodes": update.get("episodes") or [],
        "observations": update.get("observations") or [],
    }


def ingest(*args: Any, **kwargs: Any) -> tuple[Passport, list[ResolveStats]]:
    """
    Deprecated. Use SozoGraph().ingest(), which returns the Passport directly.

    Kept for one release so 0.1.1 callers expecting a (passport, stats) tuple
    keep working.
    """
    import warnings

    warnings.warn(
        "sozograph.core.ingest() is deprecated; use SozoGraph().ingest(), which "
        "returns a Passport with .stats attached.",
        DeprecationWarning,
        stacklevel=2,
    )
    sg = SozoGraph()
    passport = sg.ingest(*args, **kwargs)
    return passport, list(getattr(passport, "stats", []))
