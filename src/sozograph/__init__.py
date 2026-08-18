"""
SozoGraph: portable JSON memory for LLM agents.

    from sozograph import SozoGraph

    sg = SozoGraph()
    passport = sg.ingest(transcript)
    print(passport.context(query="Where does Melanie live?"))
    passport.save("user.json")

No vector database, no embedding model, no local weights. The memory is a small
JSON file you can read, diff, email, and load anywhere.
"""
from __future__ import annotations

from .compact import compact
from .core import SozoGraph
from .providers import LLMProvider, ProviderError, Usage, get_provider
from .schema import (
    Contradiction,
    Entity,
    Episode,
    Fact,
    OpenLoop,
    Passport,
    Preference,
    SourceRef,
)

__all__ = [
    "SozoGraph",
    "Passport",
    "Episode",
    "Fact",
    "Preference",
    "Entity",
    "OpenLoop",
    "Contradiction",
    "SourceRef",
    "compact",
    "get_provider",
    "LLMProvider",
    "ProviderError",
    "Usage",
]

__version__ = "0.2.0"
