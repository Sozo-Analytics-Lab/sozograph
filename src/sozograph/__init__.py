"""
SozoGraph: the portable memory layer for agents.

    from sozograph import Passport

    p = Passport.issue("rairo")                       # one holder, one passport
    muse = p.issue_visa("muse", domains=["shopping"], write=True)
    p.propose([{"attribute": "shoe_size", "value": "44", "origin": "user_direct"}], visa=muse)
    print(p.view(muse, query="running shoes").context)
    p.save("rairo.json")

The holder owns the passport. Each agent holds a visa. Every write arrives as
a stamp. The core needs only pydantic: no vector database, no embedding model,
no service. A passport is a small JSON file you can read, diff, merge and
carry between agents.
"""
from __future__ import annotations

from .compact import compact
from .core import SozoGraph
from .protocol import ProposeResult
from .providers import LLMProvider, ProviderError, Usage, get_provider
from .recall import RecallRecord, RecallResult
from .schema import (
    AgentRecord,
    Contradiction,
    Entity,
    Episode,
    EvidenceRef,
    Fact,
    Observation,
    OpenLoop,
    Passport,
    Preference,
    SourceRef,
    Stamp,
    Visa,
)
from .stamps import HmacSigner

__all__ = [
    "SozoGraph",
    "Passport",
    "Visa",
    "Stamp",
    "AgentRecord",
    "ProposeResult",
    "HmacSigner",
    "Episode",
    "Fact",
    "Preference",
    "Entity",
    "Observation",
    "OpenLoop",
    "Contradiction",
    "SourceRef",
    "EvidenceRef",
    "RecallRecord",
    "RecallResult",
    "compact",
    "get_provider",
    "LLMProvider",
    "ProviderError",
    "Usage",
]

__version__ = "0.4.0"
