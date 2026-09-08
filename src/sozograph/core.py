from __future__ import annotations

import os
from typing import Any

from .batching import DEFAULT_MAX_TOKENS
from .batching import plan as plan_batches
from .ingest import coerce_to_interactions, load_ingest_config
from .providers import LLMProvider, from_env, get_provider
from .render import export_context as _export_context
from .resolver import ResolveStats
from .schema import Passport


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
        retention: str = "none",
        extraction_revision: str = "0.3.2",
        reextract: bool = False,
        max_extra_calls: int = 8,
        max_split_depth: int = 2,
        clock=None,
        checkpoint=None,
    ) -> Passport:
        """
        Ingest a transcript, a database object, or a list of either.

        Returns the updated Passport. Merge statistics are on `passport.stats`.

        Interactions are batched into token-bounded segments by default, one
        extraction call each. Pass `batch=False` to extract per interaction,
        which costs one call per turn and is almost never what you want.
        """
        from .ingestion_engine import ingest_into
        return ingest_into(self, data, passport=passport, meta=meta, hint=hint, batch=batch,
                           max_segment_tokens=max_segment_tokens, retention=retention,
                           extraction_revision=extraction_revision, reextract=reextract,
                           max_extra_calls=max_extra_calls, max_split_depth=max_split_depth,
                           clock=clock, checkpoint=checkpoint)

    def plan(
        self,
        data: Any,
        *,
        meta: dict[str, Any] | None = None,
        hint: str | None = None,
        max_segment_tokens: int = DEFAULT_MAX_TOKENS,
        max_extra_calls: int = 8,
        retention: str = "none",
    ) -> dict[str, float]:
        """
        Report what ingesting `data` will cost, without calling a model.

        Worth running once before ingesting a long history.
        """
        interactions, _ = coerce_to_interactions(data, hint=hint, meta=meta or {})
        from .ingestion_engine import normalize_database_text
        normalize_database_text(interactions)
        report = plan_batches(interactions, max_tokens=max_segment_tokens)
        if retention not in {"none", "excerpts", "full"} or max_extra_calls < 0:
            raise ValueError("Invalid retention or call bound")
        from .prompts import EXTRACTOR_SYSTEM_PROMPT, EXTRACTOR_USER_PROMPT_TEMPLATE
        overhead = len(EXTRACTOR_SYSTEM_PROMPT) + len(EXTRACTOR_USER_PROMPT_TEMPLATE) + 100
        report["estimated_prompt_overhead_tokens"] = int(overhead * report["segments"] / 3.6)
        report["estimated_input_tokens"] += report["estimated_prompt_overhead_tokens"]
        report["max_extraction_calls"] = report["segments"] + max_extra_calls
        report["source_text_bytes"] = sum(len(i.text.encode("utf-8")) for i in interactions)
        report["original_source_text_bytes_before_overlap"] = report["source_text_bytes"] if retention != "none" else 0
        return report

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
            budget_chars=_default_context_budget() if budget_chars is None else budget_chars,
            header=header,
        )


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
