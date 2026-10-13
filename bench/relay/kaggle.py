"""
Kaggle Benchmarks agents for Passport Relay.

Locally, `kaggle benchmarks auth -y` writes model proxy credentials to `.env`;
on the platform the task receives its model. Either way an agent is one
isolated chat per call, with output capped so the proxy reserves little quota.
"""
from __future__ import annotations

import re
import threading
import time

MAX_TOKENS = 16000


class CostMeter:
    def __init__(self):
        self.nanodollars = 0
        self.calls = 0
        self.failures = 0
        self._lock = threading.Lock()

    def add(self, nanodollars: int) -> None:
        with self._lock:
            self.nanodollars += int(nanodollars or 0)
            self.calls += 1

    @property
    def usd(self) -> float:
        return self.nanodollars / 1e9


def kaggle_agent(llm, meter: CostMeter, label: str = "relay"):
    """Wrap a kaggle_benchmarks LLM as a Relay agent: (system, user) -> reply text."""
    import kaggle_benchmarks as kbench

    def once(system: str, user: str) -> str:
        last = None
        for params in ({"max_tokens": MAX_TOKENS}, None):
            try:
                with kbench.chats.new(label, system_instructions=system, orphan=True) as chat:
                    reply = llm.prompt(user, extra_api_params=params)
                meter.add(chat.usage.total_cost_nanodollars or 0)
                return str(reply)
            except Exception as exc:  # retry once without the cap when a backend rejects it
                last = exc
                if params and re.search(r"max_tokens|unexpected keyword|unknown (field|parameter)", str(exc), re.I):
                    continue
                raise
        raise last

    def ask(system: str, user: str) -> str:
        for attempt in range(3):
            try:
                return once(system, user)
            except Exception:
                if attempt == 2:
                    # A session the agent could not answer writes nothing; the run continues.
                    meter.failures += 1
                    return '```json\n{"stamps": []}\n```'
                time.sleep(5 * (attempt + 1))
        return '```json\n{"stamps": []}\n```'

    return ask


def local_llm(name: str):
    import kaggle_benchmarks as kbench
    from kaggle_benchmarks import _config, events
    if _config.config.ui_handler is not None:
        events.manager.unbind(_config.config.ui_handler)
    return kbench.llms[name]
