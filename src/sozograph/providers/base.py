from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


class ProviderError(RuntimeError):
    """Raised when a provider cannot fulfil a request."""


class MissingDependency(ProviderError):
    """Raised when a provider's optional SDK is not installed."""

    def __init__(self, module: str, extra: str):
        super().__init__(
            f"The '{module}' package is required for this provider.\n"
            f"Install it with:  pip install 'sozograph[{extra}]'"
        )
        self.module = module
        self.extra = extra


def require(module: str, extra: str):
    """
    Import an optional SDK, or explain exactly how to install it.

    Every provider SDK is imported through here at call time, never at module
    import time. That is what keeps `pip install sozograph` a pydantic-only
    install and lets a passport be loaded and rendered with no SDK present.
    """
    try:
        return __import__(module, fromlist=["__name__"])
    except ImportError as exc:  # pragma: no cover - exercised via tests with fakes
        raise MissingDependency(module, extra) from exc


@dataclass
class Usage:
    """
    Token and call accounting.

    This is not diagnostic decoration. The benchmark table's token and API-call
    columns are these counters, so every provider must populate them.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt_tokens: int = 0, completion_tokens: int = 0, calls: int = 1) -> None:
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        self.calls += int(calls)

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            calls=self.calls + other.calls,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def loads_lenient(text: str) -> dict[str, Any]:
    """
    Parse JSON from a model response.

    Providers with native schema enforcement return clean JSON. This exists for
    the ones that can only be asked nicely (Ollama with a small local model,
    older OpenAI-compatible gateways), which sometimes wrap output in a code
    fence or add a trailing sentence.
    """
    if text is None:
        raise ProviderError("Model returned no content")
    s = str(text).strip()
    if not s:
        raise ProviderError("Model returned empty content")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    stripped = _FENCE_RE.sub("", s).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Last resort: the outermost {...} span.
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ProviderError(f"Model did not return valid JSON. Got: {s[:400]}")


@dataclass
class LLMProvider(ABC):
    """
    The whole contract the core engine knows about.

    Everything provider-shaped lives below this line: tool definitions, JSON
    mode, grammar-constrained decoding, transport, retries. The engine above it
    only ever asks for "JSON matching this schema" or "some text".
    """

    model: str
    api_key: str | None = None
    timeout: float = 120.0
    max_retries: int = 2
    usage: Usage = field(default_factory=Usage)

    #: Short provider name, e.g. "openai". ClassVar so it is not an init arg.
    name: ClassVar[str] = "base"

    @abstractmethod
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Return a dict conforming to `schema`, enforced natively where possible."""

    @abstractmethod
    def complete_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
    ) -> str:
        """Return free text."""

    def reset_usage(self) -> None:
        self.usage = Usage()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} model={self.model!r} calls={self.usage.calls}>"
