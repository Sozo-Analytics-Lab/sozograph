from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from .base import LLMProvider, ProviderError, loads_lenient


@dataclass
class LangChainProvider(LLMProvider):
    """
    Bring your own LangChain chat model.

    Pass an already-configured BaseChatModel and SozoGraph drives it, so an
    existing LangChain setup (callbacks, caching, tracing, custom retries)
    carries over untouched.

        from langchain_openai import ChatOpenAI
        SozoGraph(provider=LangChainProvider(chat_model=ChatOpenAI(model="gpt-4o-mini")))
    """

    name: ClassVar[str] = "langchain"
    model: str = "langchain"
    chat_model: Any | None = None

    def __post_init__(self):
        if self.chat_model is None:
            raise ProviderError(
                "LangChainProvider requires chat_model=<a BaseChatModel instance>."
            )
        # Prefer the wrapped model's own identifier for reporting.
        for attr in ("model_name", "model", "model_id"):
            value = getattr(self.chat_model, attr, None)
            if isinstance(value, str) and value:
                self.model = value
                break

    def _record(self, msg) -> None:
        u = getattr(msg, "usage_metadata", None) or {}
        get = u.get if isinstance(u, dict) else lambda k, d=0: getattr(u, k, d)
        self.usage.add(get("input_tokens", 0) or 0, get("output_tokens", 0) or 0)

    def _messages(self, system: str, user: str):
        return [("system", system), ("human", user)]

    def complete_json(self, *, system, user, schema, temperature=0.2) -> dict[str, Any]:
        try:
            structured = self.chat_model.with_structured_output(schema)
            result = structured.invoke(self._messages(system, user))
            # with_structured_output hides raw usage; count the call regardless.
            self.usage.add(0, 0)
            if isinstance(result, dict):
                return result
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return loads_lenient(str(result))
        except (AttributeError, NotImplementedError):
            msg = self.chat_model.invoke(self._messages(system, user))
            self._record(msg)
            return loads_lenient(getattr(msg, "content", msg))

    def complete_text(self, *, system, user, temperature=0.2) -> str:
        msg = self.chat_model.invoke(self._messages(system, user))
        self._record(msg)
        content = getattr(msg, "content", msg)
        return str(content).strip()
