from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar

from .base import LLMProvider, loads_lenient, require

# Gemini's response_schema is an OpenAPI 3.0 subset and rejects these.
_UNSUPPORTED = ("additionalProperties", "$schema", "strict", "definitions", "$defs")

# Free-tier quota (RESOURCE_EXHAUSTED, HTTP 429) is routine, not exceptional:
# Google's own free tier caps requests per minute per model, so a multi-call
# benchmark run hits it by design, not by mistake. Wait out the server's own
# suggested delay and retry, rather than let the run die on the first 429.
_RATE_LIMIT_CODE = 429
_MAX_RATE_LIMIT_RETRIES = 8
_DEFAULT_RETRY_DELAY_SECONDS = 5.0


def _retry_delay_seconds(exc: Exception) -> float:
    """Read the server-suggested retry delay out of a 429's error body."""
    details = getattr(exc, "details", None)
    items = []
    if isinstance(details, dict):
        items = (details.get("error") or {}).get("details") or []
    for item in items:
        if isinstance(item, dict) and "retryDelay" in item:
            try:
                return float(str(item["retryDelay"]).rstrip("s"))
            except ValueError:
                continue
    return _DEFAULT_RETRY_DELAY_SECONDS


def _to_gemini_schema(node: Any) -> Any:
    """Strip JSON Schema keywords Gemini's response_schema does not accept."""
    if isinstance(node, dict):
        return {k: _to_gemini_schema(v) for k, v in node.items() if k not in _UNSUPPORTED}
    if isinstance(node, list):
        return [_to_gemini_schema(v) for v in node]
    return node


@dataclass
class GeminiProvider(LLMProvider):
    """
    Google Gemini.

    Two things 0.1.1 got wrong are corrected here: the system prompt goes to
    `system_instruction` (Gemini's content roles are user/model, so a
    role="system" entry was never a system prompt), and the JSON schema goes to
    `response_schema` instead of being pasted into the prompt as prose.
    """

    name: ClassVar[str] = "gemini"

    def _client(self):
        if getattr(self, "_cached", None) is None:
            genai = require("google.genai", "gemini")
            self._cached = genai.Client(api_key=self.api_key)
        return self._cached

    def _types(self):
        return require("google.genai.types", "gemini")

    def _record(self, resp) -> None:
        u = getattr(resp, "usage_metadata", None)
        self.usage.add(
            getattr(u, "prompt_token_count", 0) or 0,
            getattr(u, "candidates_token_count", 0) or 0,
        )

    def _generate(self, *, contents, config):
        attempt = 0
        while True:
            try:
                return self._client().models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            except Exception as exc:
                if getattr(exc, "code", None) != _RATE_LIMIT_CODE or (
                    attempt >= _MAX_RATE_LIMIT_RETRIES
                ):
                    raise
                time.sleep(_retry_delay_seconds(exc))
                attempt += 1

    def complete_json(self, *, system, user, schema, temperature=0.2) -> dict[str, Any]:
        types = self._types()
        resp = self._generate(
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=_to_gemini_schema(schema),
            ),
        )
        self._record(resp)
        return loads_lenient(resp.text)

    def complete_text(self, *, system, user, temperature=0.2) -> str:
        types = self._types()
        resp = self._generate(
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
            ),
        )
        self._record(resp)
        return (resp.text or "").strip()
