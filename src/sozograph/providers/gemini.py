from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar

from .base import LLMProvider, loads_lenient, require

# Gemini's response_schema is an OpenAPI 3.0 subset and rejects these.
_UNSUPPORTED = ("additionalProperties", "$schema", "strict", "definitions", "$defs")

# Two error codes are routine under real load, not exceptional: 429
# (RESOURCE_EXHAUSTED) is Google's free-tier per-minute cap, hit by design on
# any multi-call benchmark run; 503 (UNAVAILABLE) is transient overload on
# Google's side, unrelated to anything the caller did. Both are worth waiting
# out and retrying rather than letting the run die on the first one.
_RETRYABLE_CODES = (429, 503)
# A single 429 with a ~60s suggested delay should almost always clear on the
# next attempt; 8 retries turned out to be too few during a sustained
# throttled stretch (a burst of prior calls keeping the rolling window full).
# Correctness matters more than speed for an unattended multi-hour run, so
# this affords up to ~30 minutes of waiting on one call before giving up.
_MAX_RATE_LIMIT_RETRIES = 30
_DEFAULT_RETRY_DELAY_SECONDS = 5.0
_BACKOFF_BASE_SECONDS = 10.0
_BACKOFF_MAX_SECONDS = 60.0


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    """
    Prefer the server-suggested delay (429 carries a RetryInfo with one); fall
    back to exponential backoff for errors that don't (503 usually doesn't).
    """
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
    if getattr(exc, "code", None) == 429:
        return _DEFAULT_RETRY_DELAY_SECONDS
    return min(_BACKOFF_BASE_SECONDS * (2**attempt), _BACKOFF_MAX_SECONDS)


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
                if getattr(exc, "code", None) not in _RETRYABLE_CODES or (
                    attempt >= _MAX_RATE_LIMIT_RETRIES
                ):
                    raise
                time.sleep(_retry_delay_seconds(exc, attempt))
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
