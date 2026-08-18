from __future__ import annotations

import os
from typing import Any

from .batching import DEFAULT_MAX_TOKENS, segment_interactions
from .batching import plan as plan_batches
from .extractor import Extractor
from .ingest import apply_fallback_summaries, coerce_to_interactions, load_ingest_config
from .providers import LLMProvider, from_env, get_provider
from .render import export_context as _export_context
from .resolver import ResolveStats, merge_passport_update
from .schema import Passport, SourceRef
from .utils import sha256_json, stable_id


def _default_context_budget() -> int:
    try:
        return int(os.getenv("SOZOGRAPH_DEFAULT_CONTEXT_BUDGET", "3000"))
    except ValueError:
        return 3000


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
                update = extractor.extract_segment(
                    segment, known_keys=base.known_keys()
                )
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


def _update_kwargs(update: dict[str, Any]) -> dict[str, Any]:
    return {
        "facts": update.get("facts") or [],
        "prefs": update.get("prefs") or [],
        "entities": update.get("entities") or [],
        "open_loops": update.get("open_loops") or [],
        "episodes": update.get("episodes") or [],
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
