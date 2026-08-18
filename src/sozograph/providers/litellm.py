from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from .base import LLMProvider, loads_lenient, require


@dataclass
class LiteLLMProvider(LLMProvider):
    """
    Anything LiteLLM can reach, which is roughly a hundred providers.

    Model strings pass straight through in LiteLLM's own notation, for example
    "bedrock/anthropic.claude-3-5-sonnet" or "vertex_ai/gemini-2.0-flash".
    """

    name: ClassVar[str] = "litellm"
    base_url: str | None = None

    def _lib(self):
        if getattr(self, "_cached", None) is None:
            self._cached = require("litellm", "litellm")
        return self._cached

    def _kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"timeout": self.timeout, "num_retries": self.max_retries}
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.base_url:
            kw["api_base"] = self.base_url
        return kw

    def _record(self, resp) -> None:
        u = getattr(resp, "usage", None) or {}
        get = u.get if isinstance(u, dict) else lambda k, d=0: getattr(u, k, d)
        self.usage.add(get("prompt_tokens", 0) or 0, get("completion_tokens", 0) or 0)

    def complete_json(self, *, system, user, schema, temperature=0.2) -> dict[str, Any]:
        resp = self._lib().completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "memory_update", "strict": True, "schema": schema},
            },
            **self._kwargs(),
        )
        self._record(resp)
        return loads_lenient(resp.choices[0].message.content)

    def complete_text(self, *, system, user, temperature=0.2) -> str:
        resp = self._lib().completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            **self._kwargs(),
        )
        self._record(resp)
        return (resp.choices[0].message.content or "").strip()
