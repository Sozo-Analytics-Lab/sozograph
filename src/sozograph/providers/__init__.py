from __future__ import annotations

import os
from typing import Any

from .base import LLMProvider, MissingDependency, ProviderError, Usage

__all__ = [
    "LLMProvider",
    "ProviderError",
    "MissingDependency",
    "Usage",
    "get_provider",
    "from_env",
    "register_provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "OllamaProvider",
    "LiteLLMProvider",
    "LangChainProvider",
]

# Sensible default per provider. Every one is overridable with model=...
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.2",
    "litellm": "gpt-4o-mini",
}

# Provider name -> (module, class). Imported lazily so that neither the
# provider module nor its SDK is touched until someone actually asks for it.
_REGISTRY: dict[str, tuple] = {
    "anthropic": (".anthropic", "AnthropicProvider"),
    "openai": (".openai", "OpenAIProvider"),
    "gemini": (".gemini", "GeminiProvider"),
    "google": (".gemini", "GeminiProvider"),
    "ollama": (".ollama", "OllamaProvider"),
    "litellm": (".litellm", "LiteLLMProvider"),
    "langchain": (".langchain", "LangChainProvider"),
}

# Which env var carries the key for each provider.
_ENV_KEYS: dict[str, tuple] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "litellm": (),
    "ollama": (),
}


def _load(name: str) -> type[LLMProvider]:
    import importlib

    try:
        module_path, class_name = _REGISTRY[name]
    except KeyError:
        raise ProviderError(
            f"Unknown provider {name!r}. Available: {', '.join(sorted(set(_REGISTRY)))}"
        ) from None
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, class_name)


def register_provider(name: str, module_path: str, class_name: str) -> None:
    """Register a third-party provider so get_provider() can resolve it."""
    _REGISTRY[name.lower().strip()] = (module_path, class_name)


def get_provider(
    spec: str | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Build a provider from a spec string.

        get_provider("openai")                    -> OpenAI with the default model
        get_provider("openai:gpt-4o-mini")        -> explicit model
        get_provider("anthropic", model="...")    -> same, via kwarg
        get_provider("ollama:llama3.2", host=...) -> extra kwargs pass through

    The colon form keeps a whole configuration in one string, which is what
    makes the benchmark harness and env-var configuration a single field.
    """
    if not spec:
        return from_env(model=model, api_key=api_key, **kwargs)

    name, _, spec_model = spec.partition(":")
    name = name.lower().strip()
    if name not in _REGISTRY:
        raise ProviderError(
            f"Unknown provider {name!r}. Available: {', '.join(sorted(set(_REGISTRY)))}"
        )
    resolved_model = model or spec_model or DEFAULT_MODELS.get(name)
    if not resolved_model:
        raise ProviderError(f"No model given for provider {name!r} and no default is known.")

    if api_key is None:
        for env_name in _ENV_KEYS.get(name, ()):
            api_key = os.getenv(env_name)
            if api_key:
                break

    return _load(name)(model=resolved_model, api_key=api_key, **kwargs)


def from_env(**kwargs: Any) -> LLMProvider:
    """
    Pick a provider from the environment so SozoGraph() just works.

    Order: SOZOGRAPH_PROVIDER, then whichever vendor key is set, then a local
    Ollama if one is reachable.
    """
    spec = os.getenv("SOZOGRAPH_PROVIDER")
    if spec:
        model = kwargs.pop("model", None) or os.getenv("SOZOGRAPH_MODEL")
        return get_provider(spec, model=model, **kwargs)

    for name in ("anthropic", "openai", "gemini"):
        if any(os.getenv(v) for v in _ENV_KEYS[name]):
            model = kwargs.pop("model", None) or os.getenv("SOZOGRAPH_MODEL")
            return get_provider(name, model=model, **kwargs)

    if _ollama_is_reachable():
        return get_provider("ollama", **kwargs)

    raise ProviderError(
        "No LLM provider configured.\n"
        "Set one of ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY, "
        "or SOZOGRAPH_PROVIDER=<name>[:<model>], "
        "or run a local Ollama, "
        "or pass provider=... to SozoGraph()."
    )


def _ollama_is_reachable(timeout: float = 0.35) -> bool:
    import urllib.error
    import urllib.request

    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not host.startswith("http"):
        host = f"http://{host}"
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def __getattr__(name: str) -> Any:
    # Keep `from sozograph.providers import OpenAIProvider` working without
    # importing all six provider modules (and their SDKs) at package import.
    for key, (_, class_name) in _REGISTRY.items():
        if class_name == name:
            return _load(key)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
