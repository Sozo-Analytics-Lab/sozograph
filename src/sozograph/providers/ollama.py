from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar

from .base import LLMProvider, loads_lenient, require


@dataclass
class OllamaProvider(LLMProvider):
    """
    Local models via Ollama.

    Ollama's `format` parameter accepts a JSON Schema and constrains decoding
    against it, so a 3B model on a laptop still returns valid structure. This is
    the configuration that needs no API key, no cloud, and no GPU.
    """

    name: ClassVar[str] = "ollama"
    host: str | None = None
    #: Ollama's own "vram-based default" tops out at 4096 regardless of how
    #: much VRAM is actually free -- observed truncating mid-JSON on an
    #: extraction segment well within a 16GB GPU's headroom. None keeps
    #: Ollama's default so laptop/no-GPU users aren't pushed into a bigger
    #: KV-cache than they asked for.
    num_ctx: int | None = None

    def _options(self, temperature: float) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        return options

    def _client(self):
        if getattr(self, "_cached", None) is None:
            ollama = require("ollama", "ollama")
            host = self.host or os.getenv("OLLAMA_HOST")
            self._cached = ollama.Client(host=host) if host else ollama.Client()
        return self._cached

    def _record(self, resp: dict[str, Any]) -> None:
        get = resp.get if isinstance(resp, dict) else lambda k, d=0: getattr(resp, k, d)
        self.usage.add(get("prompt_eval_count", 0) or 0, get("eval_count", 0) or 0)

    def _content(self, resp: Any) -> str:
        if isinstance(resp, dict):
            return (resp.get("message") or {}).get("content", "")
        return getattr(getattr(resp, "message", None), "content", "") or ""

    def complete_json(self, *, system, user, schema, temperature=0.2) -> dict[str, Any]:
        resp = self._client().chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=schema,
            options=self._options(temperature),
        )
        self._record(resp)
        return loads_lenient(self._content(resp))

    def complete_text(self, *, system, user, temperature=0.2) -> str:
        resp = self._client().chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options=self._options(temperature),
        )
        self._record(resp)
        return self._content(resp).strip()
