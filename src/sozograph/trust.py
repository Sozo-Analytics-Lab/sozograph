"""
Competence: how much the holder can rely on each agent, per domain.

Each (agent, domain) pair holds a Beta posterior. A holder confirmation of an
agent's stamp counts as a success; a holder correction or retraction of an
accepted stamp counts as a failure. The prior is kappa pseudo-observations at
the agent model's Passport Control write score when we have measured it, and
at a home-domain or away default when we have not. Every number here is a
function of the stamp log and the agent registry, so every replica agrees.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .schema import AgentRecord, Stamp

KAPPA = 4.0
HOME_Q = 0.85
AWAY_Q = 0.60
HOLDER_Q = 0.99
HOLDER = "holder"

#: Passport Control (11 October 2026) write scores, scorer v3.1.
PASSPORT_CONTROL = {
    "claude-opus-5-5": 0.992, "gpt-6.1-sol": 0.970, "gemini-3.1-pro": 0.952,
    "claude-sonnet-5-5": 0.945, "glm-5": 0.926, "gemini-3.7-flash": 0.919,
    "gemini-3.8-flash": 0.915, "gemini-2.5-flash": 0.905, "gemma-4-31b": 0.900,
    "claude-haiku-4-5": 0.866, "deepseek-r1": 0.863, "gemini-3.5-flash-lite": 0.856,
    "gemini-3.1-flash-lite": 0.842, "qwen3-235b": 0.814, "gpt-5.4-mini": 0.790,
    "gpt-oss-20b": 0.761, "gpt-5.4-nano": 0.748, "gpt-oss-120b": 0.740,
}


def model_score(model: str) -> float | None:
    """The measured write score of a model name, matched loosely."""
    name = (model or "").lower().replace("_", "-").replace(".", "-")
    if not name:
        return None
    best, length = None, 0
    for key, score in PASSPORT_CONTROL.items():
        k = key.replace(".", "-")
        if k in name and len(k) > length:
            best, length = score, len(k)
    return best


def prior(agent: str, domain: str, agents: dict[str, AgentRecord]) -> tuple[float, float]:
    if agent == HOLDER:
        q = HOLDER_Q
    else:
        record = agents.get(agent)
        measured = model_score(record.model) if record else None
        home = set(record.home_domains) if record else set()
        if measured is not None and (not home or domain in home or domain == "general"):
            q = measured
        elif domain in home or domain == "general" or not home:
            q = HOME_Q
        else:
            q = AWAY_Q
    return KAPPA * q, KAPPA * (1.0 - q)


def evidence_counts(stamps: list[Stamp]) -> dict[tuple[str, str], list[int]]:
    """Successes and failures per (author, domain), read from holder stamps."""
    by_id = {s.id: s for s in stamps}
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for s in stamps:
        if s.author != HOLDER or s.admission != "accepted" or s.op not in {"confirm", "correct", "retract"}:
            continue
        for target_id in s.targets:
            target = by_id.get(target_id)
            if target is None or target.author == HOLDER:
                continue
            if s.op != "confirm" and target.admission != "accepted":
                # Dismissing a quarantined write is the gate working, not the agent failing.
                continue
            for domain in (target.claim.get("domains") or ["general"]):
                counts[(target.author, domain)][0 if s.op == "confirm" else 1] += 1
    return counts


class Competence:
    """Posterior means for every (agent, domain) the log mentions."""

    def __init__(self, stamps: list[Stamp], agents: dict[str, AgentRecord]):
        self.agents = agents
        self.counts = evidence_counts(stamps)

    def posterior(self, agent: str, domain: str) -> tuple[float, float]:
        a0, b0 = prior(agent, domain, self.agents)
        plus, minus = self.counts.get((agent, domain), [0, 0])
        return a0 + plus, b0 + minus

    def mean(self, agent: str, domain: str) -> float:
        a, b = self.posterior(agent, domain)
        return a / (a + b)

    def logit(self, agent: str, domain: str) -> float:
        p = min(max(self.mean(agent, domain), 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    def table(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        keys = set(self.counts) | {(a, d) for a, r in self.agents.items() for d in (r.home_domains or ["general"])}
        for agent, domain in sorted(keys):
            out.setdefault(agent, {})[domain] = round(self.mean(agent, domain), 4)
        return out
