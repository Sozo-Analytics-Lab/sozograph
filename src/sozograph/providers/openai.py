from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, ClassVar

from .base import LLMProvider, loads_lenient, require

# OpenAI-compatible gateways (Groq in particular) throttle on tokens-per-minute
# as much as requests-per-minute, and hitting that under real load is routine,
# not exceptional. Their 429 bodies say "try again in <n>ms" rather than
# carrying a structured retry-after value, so that's what gets parsed.
_RETRYABLE_STATUS_CODES = (429, 503)
_MAX_RATE_LIMIT_RETRIES = 30
_DEFAULT_RETRY_DELAY_SECONDS = 5.0
_TRY_AGAIN_MS_RE = re.compile(r"try again in\s+([\d.]+)\s*ms", re.IGNORECASE)
_TRY_AGAIN_S_RE = re.compile(r"try again in\s+([\d.]+)\s*s\b", re.IGNORECASE)


def _retry_delay_seconds(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}).get("retry-after") if response else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass

    message = str(exc)
    if m := _TRY_AGAIN_MS_RE.search(message):
        return float(m.group(1)) / 1000.0
    if m := _TRY_AGAIN_S_RE.search(message):
        return float(m.group(1))
    return _DEFAULT_RETRY_DELAY_SECONDS


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

    def _create(self, **kwargs):
        attempt = 0
        while True:
            try:
                return self._client().chat.completions.create(**kwargs)
            except Exception as exc:
                code = getattr(exc, "status_code", None)
                if code not in _RETRYABLE_STATUS_CODES or attempt >= _MAX_RATE_LIMIT_RETRIES:
                    raise
                time.sleep(_retry_delay_seconds(exc))
                attempt += 1

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
            resp = self._create(**kwargs)
        except Exception as exc:
            # Gateways and reasoning models vary in what they accept -- and in
            # how reliably they honor strict mode at all. Groq's gpt-oss holds
            # to it; qwen3.6 on the same gateway sometimes emits an empty
            # completion under it, which Groq itself then rejects as invalid
            # JSON (code "json_validate_failed", no "json_schema" substring in
            # sight). Degrade to plain JSON mode rather than failing the whole
            # ingestion on either kind of complaint.
            msg = str(exc).lower()
            if "response_format" in msg or "json" in msg or "temperature" in msg:
                kwargs["response_format"] = {"type": "json_object"}
                kwargs.pop("temperature", None)
                resp = self._create(**kwargs)
            else:
                raise
        self._record(resp)
        return loads_lenient(resp.choices[0].message.content)

    def complete_text(self, *, system, user, temperature=0.2) -> str:
        resp = self._create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        self._record(resp)
        return (resp.choices[0].message.content or "").strip()
