from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from .base import LLMProvider, ProviderError, require

_TOOL_NAME = "emit_memory_update"


@dataclass
class AnthropicProvider(LLMProvider):
    """
    Anthropic via a forced tool call.

    A tool definition carries a real JSON Schema and `tool_choice` makes the
    model use it, so the response is structurally guaranteed rather than
    requested in prose. This is the strictest of the six.
    """

    name: ClassVar[str] = "anthropic"
    max_tokens: int = 8192

    def _client(self):
        if getattr(self, "_cached", None) is None:
            anthropic = require("anthropic", "anthropic")
            self._cached = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._cached

    def _record(self, resp) -> None:
        u = getattr(resp, "usage", None)
        self.usage.add(
            getattr(u, "input_tokens", 0) or 0,
            getattr(u, "output_tokens", 0) or 0,
        )

    def complete_json(self, *, system, user, schema, temperature=0.2) -> dict[str, Any]:
        resp = self._client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            temperature=temperature,
            tools=[{
                "name": _TOOL_NAME,
                "description": "Record the structured memory update extracted from the text.",
                "strict": True,
                "input_schema": schema,
            }],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=[{"role": "user", "content": user}],
        )
        self._record(resp)
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise ProviderError("Anthropic returned no tool_use block")

    def complete_text(self, *, system, user, temperature=0.2) -> str:
        resp = self._client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        self._record(resp)
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip()
