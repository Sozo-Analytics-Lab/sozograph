from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from .base import LLMProvider, loads_lenient, require


@dataclass
class OpenAIProvider(LLMProvider):
    """
    OpenAI, and every OpenAI-compatible endpoint.

    Structured output uses `response_format` with a strict JSON schema, which
    constrains decoding rather than asking the model to behave. Point
    `base_url` at vLLM, Together, Groq, OpenRouter, or DeepSeek and it works
    unchanged.
    """

    name: ClassVar[str] = "openai"
    base_url: str | None = None

    def _client(self):
        if getattr(self, "_cached", None) is None:
            openai = require("openai", "openai")
            kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._cached = openai.OpenAI(**kwargs)
        return self._cached

    def _record(self, resp) -> None:
        u = getattr(resp, "usage", None)
        self.usage.add(
            getattr(u, "prompt_tokens", 0) or 0,
            getattr(u, "completion_tokens", 0) or 0,
        )

    def complete_json(self, *, system, user, schema, temperature=0.2) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "memory_update",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        try:
            resp = self._client().chat.completions.create(**kwargs)
        except Exception as exc:
            # Gateways and reasoning models vary in what they accept. Degrade to
            # plain JSON mode rather than failing the whole ingestion.
            msg = str(exc).lower()
            if "response_format" in msg or "json_schema" in msg or "temperature" in msg:
                kwargs["response_format"] = {"type": "json_object"}
                kwargs.pop("temperature", None)
                resp = self._client().chat.completions.create(**kwargs)
            else:
                raise
        self._record(resp)
        return loads_lenient(resp.choices[0].message.content)

    def complete_text(self, *, system, user, temperature=0.2) -> str:
        resp = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        self._record(resp)
        return (resp.choices[0].message.content or "").strip()
