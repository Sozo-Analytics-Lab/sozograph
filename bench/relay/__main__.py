"""
Run Passport Relay.

    python -m bench.relay --scripted door --scenarios 16            # offline: the gate against a door policy
    python -m bench.relay --chat anthropic:claude-opus-5-5 --inbox openai:gpt-6.1-sol \\
                          --web gemini:gemini-3.1-pro --scenarios 16 --out relay.json   # live, paid

Live runs call providers through SozoGraph's provider layer and cost money; agree a budget first.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import pc_core as pc
from .relay import door_agent, run_scenario, silent_agent, summarize


def provider_agent(spec: str):
    from sozograph.providers import get_provider
    provider = get_provider(spec)

    def ask(system: str, user: str) -> str:
        return provider.complete_text(system=system, user=user, temperature=0.0)

    return ask


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.relay")
    ap.add_argument("--scenarios", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--scripted", choices=["door", "silent"])
    ap.add_argument("--chat")
    ap.add_argument("--inbox")
    ap.add_argument("--web")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    suite = pc.make_suite(args.scenarios, args.seed)
    if args.scripted:
        policy = door_agent if args.scripted == "door" else silent_agent
        agents = {"chat": silent_agent, "inbox": policy, "web": policy}
        report = {}
        for gate in (False, True):
            results = [run_scenario(sc, agents, gate=gate) for sc in suite]
            report["gate_on" if gate else "gate_off"] = summarize(results)
        reject = {k: v["categories"].get("reject/injection") for k, v in report.items()}
        report["reject_injection"] = reject
    else:
        specs = {"chat": args.chat, "inbox": args.inbox, "web": args.web}
        if not any(specs.values()):
            ap.error("give --scripted or at least one of --chat, --inbox, --web")
        agents = {name: provider_agent(spec) for name, spec in specs.items() if spec}
        results = [run_scenario(sc, agents) for sc in suite]
        report = summarize(results)
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
