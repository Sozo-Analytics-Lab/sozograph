"""
Provider layer tests against fake transports.

No SDK is installed and no network is touched. What each test asserts is that
the provider hands the JSON schema to the engine's *native* structured-output
mechanism rather than pasting it into the prompt, and that it reports usage.
"""
from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from sozograph.providers import ProviderError, get_provider
from sozograph.providers.base import Usage, loads_lenient

SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
    "additionalProperties": False,
}
PAYLOAD = {"facts": ["lives in Kwekwe"]}


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


@pytest.fixture
def captured() -> dict[str, Any]:
    return {}


# --------------------------------------------------------------------------
# Anthropic: forced tool_choice + strict input_schema
# --------------------------------------------------------------------------

@pytest.fixture
def fake_anthropic(monkeypatch, captured):
    class Block:
        type = "tool_use"
        input = PAYLOAD

    class TextBlock:
        type = "text"
        text = "hello"

    class Usg:
        input_tokens, output_tokens = 120, 40

    class Messages:
        def create(self, **kw):
            captured.update(kw)
            content = [Block()] if "tools" in kw else [TextBlock()]
            return types.SimpleNamespace(content=content, usage=Usg())

    class Anthropic:
        def __init__(self, **kw):
            captured["client_kwargs"] = kw
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", _module("anthropic", Anthropic=Anthropic))


def test_anthropic_uses_forced_tool_choice(fake_anthropic, captured):
    p = get_provider("anthropic:claude-opus-5", api_key="k")
    assert p.complete_json(system="s", user="u", schema=SCHEMA) == PAYLOAD

    tool = captured["tools"][0]
    assert tool["input_schema"] is SCHEMA, "schema must reach the tool definition"
    assert tool["strict"] is True, "strict guarantees input validates against the schema"
    assert captured["tool_choice"] == {"type": "tool", "name": tool["name"]}
    assert captured["system"] == "s", "system prompt is a top-level field, not a message"
    assert p.usage.prompt_tokens == 120 and p.usage.completion_tokens == 40
    assert p.usage.calls == 1


def test_anthropic_complete_text(fake_anthropic):
    p = get_provider("anthropic", api_key="k")
    assert p.complete_text(system="s", user="u") == "hello"
    assert p.usage.calls == 1


# --------------------------------------------------------------------------
# OpenAI: response_format json_schema with strict
# --------------------------------------------------------------------------

@pytest.fixture
def fake_openai(monkeypatch, captured):
    class Completions:
        def create(self, **kw):
            captured.update(kw)
            msg = types.SimpleNamespace(content=json.dumps(PAYLOAD))
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=msg)],
                usage=types.SimpleNamespace(prompt_tokens=200, completion_tokens=30),
            )

    class OpenAI:
        def __init__(self, **kw):
            captured["client_kwargs"] = kw
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", _module("openai", OpenAI=OpenAI))


def test_openai_uses_strict_json_schema(fake_openai, captured):
    p = get_provider("openai:gpt-4o-mini", api_key="k")
    assert p.complete_json(system="s", user="u", schema=SCHEMA) == PAYLOAD

    rf = captured["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] is SCHEMA
    assert p.usage.prompt_tokens == 200 and p.usage.completion_tokens == 30


def test_openai_base_url_reaches_the_client(fake_openai, captured):
    # This is what makes vLLM / Together / Groq / OpenRouter work unchanged.
    get_provider("openai:x", api_key="k", base_url="http://localhost:8000/v1").complete_json(
        system="s", user="u", schema=SCHEMA
    )
    assert captured["client_kwargs"]["base_url"] == "http://localhost:8000/v1"


# --------------------------------------------------------------------------
# Gemini: system_instruction + native response_schema
# --------------------------------------------------------------------------

@pytest.fixture
def fake_gemini(monkeypatch, captured):
    class Models:
        def generate_content(self, **kw):
            captured.update(kw)
            return types.SimpleNamespace(
                text=json.dumps(PAYLOAD),
                usage_metadata=types.SimpleNamespace(
                    prompt_token_count=90, candidates_token_count=25
                ),
            )

    class Client:
        def __init__(self, **kw):
            captured["client_kwargs"] = kw
            self.models = Models()

    def GenerateContentConfig(**kw):
        return types.SimpleNamespace(**kw)

    google = _module("google")
    genai = _module("google.genai", Client=Client)
    gtypes = _module("google.genai.types", GenerateContentConfig=GenerateContentConfig)
    google.genai = genai
    genai.types = gtypes
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)


def test_gemini_uses_system_instruction_and_response_schema(fake_gemini, captured):
    p = get_provider("gemini:gemini-2.5-flash", api_key="k")
    assert p.complete_json(system="s", user="u", schema=SCHEMA) == PAYLOAD

    cfg = captured["config"]
    # 0.1.1 sent role="system" inside contents, which Gemini does not accept as
    # a role, and pasted the schema into the prompt as prose.
    assert cfg.system_instruction == "s"
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_schema["type"] == "object"
    assert captured["contents"] == "u"
    assert p.usage.prompt_tokens == 90 and p.usage.completion_tokens == 25


def test_gemini_schema_strips_unsupported_keywords(fake_gemini, captured):
    get_provider("gemini", api_key="k").complete_json(system="s", user="u", schema=SCHEMA)
    # response_schema is an OpenAPI 3.0 subset and rejects additionalProperties.
    assert "additionalProperties" not in captured["config"].response_schema
    assert "properties" in captured["config"].response_schema


def _retryable_error(code: int, retry_delay: str | None = "0.01s") -> Exception:
    # 429 (quota) carries a structured RetryInfo with a suggested delay; 503
    # (transient overload) typically does not, per Gemini's actual error shape.
    exc = Exception(f"{code} error")
    exc.code = code
    exc.details = {"error": {"details": []}}
    if retry_delay is not None:
        exc.details["error"]["details"] = [
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}
        ]
    return exc


@pytest.fixture
def fake_gemini_flaky(monkeypatch, captured):
    """Like fake_gemini, but generate_content fails a fixed number of times first."""

    def _install(*, failures: int, code: int = 429, retry_delay: str | None = "0.01s"):
        state = {"calls": 0}

        class Models:
            def generate_content(self, **kw):
                state["calls"] += 1
                if state["calls"] <= failures:
                    raise _retryable_error(code, retry_delay)
                captured.update(kw)
                return types.SimpleNamespace(
                    text=json.dumps(PAYLOAD),
                    usage_metadata=types.SimpleNamespace(
                        prompt_token_count=1, candidates_token_count=1
                    ),
                )

        class Client:
            def __init__(self, **kw):
                self.models = Models()

        def GenerateContentConfig(**kw):
            return types.SimpleNamespace(**kw)

        google = _module("google")
        genai = _module("google.genai", Client=Client)
        gtypes = _module("google.genai.types", GenerateContentConfig=GenerateContentConfig)
        google.genai = genai
        genai.types = gtypes
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.genai", genai)
        monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)

        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        return state, sleeps

    return _install


def test_gemini_retries_on_rate_limit_then_succeeds(fake_gemini_flaky):
    # Free-tier RPM caps make a 429 routine, not exceptional; the harness must
    # wait out the server's suggested delay and retry rather than die on it.
    state, sleeps = fake_gemini_flaky(failures=2)
    p = get_provider("gemini:gemini-2.5-flash", api_key="k")

    assert p.complete_json(system="s", user="u", schema=SCHEMA) == PAYLOAD
    assert state["calls"] == 3
    assert sleeps == [0.01, 0.01]


def test_gemini_gives_up_after_max_rate_limit_retries(fake_gemini_flaky):
    state, sleeps = fake_gemini_flaky(failures=999)
    p = get_provider("gemini:gemini-2.5-flash", api_key="k")

    with pytest.raises(Exception):
        p.complete_json(system="s", user="u", schema=SCHEMA)
    # 1 initial attempt + 30 retries, per _MAX_RATE_LIMIT_RETRIES.
    assert state["calls"] == 31
    assert len(sleeps) == 30


def test_gemini_retries_on_server_overload_then_succeeds(fake_gemini_flaky):
    # 503 UNAVAILABLE ("high demand") has no RetryInfo in the body, unlike 429;
    # this must fall back to exponential backoff rather than crash on a missing
    # retryDelay or retry with no delay at all.
    state, sleeps = fake_gemini_flaky(failures=2, code=503, retry_delay=None)
    p = get_provider("gemini:gemini-2.5-flash", api_key="k")

    assert p.complete_json(system="s", user="u", schema=SCHEMA) == PAYLOAD
    assert state["calls"] == 3
    assert sleeps == [10.0, 20.0]


# --------------------------------------------------------------------------
# Ollama: format=<schema> grammar-constrained decoding
# --------------------------------------------------------------------------

@pytest.fixture
def fake_ollama(monkeypatch, captured):
    class Client:
        def __init__(self, **kw):
            captured["client_kwargs"] = kw

        def chat(self, **kw):
            captured.update(kw)
            return {
                "message": {"content": json.dumps(PAYLOAD)},
                "prompt_eval_count": 60,
                "eval_count": 15,
            }

    monkeypatch.setitem(sys.modules, "ollama", _module("ollama", Client=Client))


def test_ollama_passes_schema_as_format(fake_ollama, captured):
    p = get_provider("ollama:llama3.2")
    assert p.complete_json(system="s", user="u", schema=SCHEMA) == PAYLOAD
    assert captured["format"] is SCHEMA, "schema constrains decoding, not the prompt"
    assert p.usage.prompt_tokens == 60 and p.usage.completion_tokens == 15


def test_ollama_needs_no_api_key(fake_ollama):
    # The laptop story: no key, no cloud.
    assert get_provider("ollama").complete_json(system="s", user="u", schema=SCHEMA) == PAYLOAD


# --------------------------------------------------------------------------
# LiteLLM and LangChain: bring-your-own transport
# --------------------------------------------------------------------------

def test_litellm_passes_through(monkeypatch, captured):
    def completion(**kw):
        captured.update(kw)
        msg = types.SimpleNamespace(content=json.dumps(PAYLOAD))
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg)],
            usage={"prompt_tokens": 11, "completion_tokens": 7},
        )

    monkeypatch.setitem(sys.modules, "litellm", _module("litellm", completion=completion))
    p = get_provider("litellm:bedrock/anthropic.claude-opus-5", api_key="k")
    assert p.complete_json(system="s", user="u", schema=SCHEMA) == PAYLOAD
    assert captured["model"] == "bedrock/anthropic.claude-opus-5"
    assert p.usage.total_tokens == 18


def test_langchain_wraps_an_existing_chat_model():
    from sozograph.providers.langchain import LangChainProvider

    class Structured:
        def invoke(self, messages):
            return dict(PAYLOAD)

    class ChatModel:
        model_name = "gpt-4o-mini"

        def with_structured_output(self, schema):
            return Structured()

        def invoke(self, messages):
            return types.SimpleNamespace(
                content="hello", usage_metadata={"input_tokens": 5, "output_tokens": 2}
            )

    p = LangChainProvider(chat_model=ChatModel())
    assert p.model == "gpt-4o-mini", "reports the wrapped model's own identifier"
    assert p.complete_json(system="s", user="u", schema=SCHEMA) == PAYLOAD
    assert p.complete_text(system="s", user="u") == "hello"
    assert p.usage.calls == 2


def test_langchain_requires_a_chat_model():
    from sozograph.providers.langchain import LangChainProvider

    with pytest.raises(ProviderError):
        LangChainProvider()


# --------------------------------------------------------------------------
# Cross-cutting
# --------------------------------------------------------------------------

def test_missing_sdk_names_the_extra_to_install(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "anthropic":
            raise ImportError("no module named anthropic")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    p = get_provider("anthropic", api_key="k")  # construction must not import
    with pytest.raises(ProviderError) as exc:
        p.complete_json(system="s", user="u", schema=SCHEMA)
    assert "sozograph[anthropic]" in str(exc.value)


def test_usage_accumulates_across_calls(fake_openai):
    p = get_provider("openai", api_key="k")
    for _ in range(3):
        p.complete_json(system="s", user="u", schema=SCHEMA)
    assert p.usage.calls == 3
    assert p.usage.total_tokens == 3 * 230
    p.reset_usage()
    assert p.usage.total_tokens == 0


@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('```\n{"a": 1}\n```', {"a": 1}),
    ('Sure! {"a": 1} hope that helps', {"a": 1}),
])
def test_lenient_json_parsing(raw, expected):
    assert loads_lenient(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "not json at all"])
def test_lenient_parsing_rejects_garbage(raw):
    with pytest.raises(ProviderError):
        loads_lenient(raw)


def test_usage_arithmetic():
    a, b = Usage(10, 5, 1), Usage(3, 2, 1)
    assert (a + b).to_dict() == {
        "prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20, "calls": 2
    }
