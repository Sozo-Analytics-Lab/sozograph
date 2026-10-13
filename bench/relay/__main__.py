"""
Run Passport Relay.

    python -m bench.relay --scripted door --scenarios 16             # offline: the gate against a door policy
    python -m bench.relay --kaggle openai/gpt-oss-120b --ablate      # Kaggle model proxy, gate on and off
    python -m bench.relay --kaggle chat=anthropic/claude-sonnet-5@default,inbox=openai/gpt-oss-120b
    python -m bench.relay --chat anthropic:claude-opus-5-5 --inbox openai:gpt-6.1-sol --web gemini:gemini-3.1-pro

Live runs spend model quota or money; agree a budget first.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time

from . import pc_core as pc
from .relay import door_agent, run_scenario, silent_agent, summarize


def provider_agent(spec: str):
    from sozograph.providers import get_provider
    provider = get_provider(spec)

    def ask(system: str, user: str) -> str:
        return provider.complete_text(system=system, user=user, temperature=0.0)

    return ask


def kaggle_agents(spec: str):
    from .kaggle import CostMeter, kaggle_agent, local_llm
    meter = CostMeter()
    if "=" in spec:
        pairs = dict(part.split("=", 1) for part in spec.split(","))
    else:
        pairs = {"chat": spec, "inbox": spec, "web": spec}
    llms = {}
    agents = {}
    for name, model in pairs.items():
        llms.setdefault(model, local_llm(model))
        agents[name] = kaggle_agent(llms[model], meter, label=f"relay-{name}")
    return agents, meter, pairs


def run_suite(suite, agents, *, gate, jobs):
    with cf.ThreadPoolExecutor(max(1, jobs)) as pool:
        return list(pool.map(lambda sc: run_scenario(sc, agents, gate=gate), suite))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.relay")
    ap.add_argument("--scenarios", type=int, default=16)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--scripted", choices=["door", "silent"])
    ap.add_argument("--kaggle", help="one kbench model for every agent, or chat=M,inbox=M,web=M")
    ap.add_argument("--ablate", action="store_true", help="also run with the gate off")
    ap.add_argument("--chat")
    ap.add_argument("--inbox")
    ap.add_argument("--web")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    suite = pc.make_suite(args.first + args.scenarios, args.seed)[args.first:]
    started = time.time()
    if args.scripted:
        policy = door_agent if args.scripted == "door" else silent_agent
        agents = {"chat": silent_agent, "inbox": policy, "web": policy}
        report = {}
        for gate in (False, True):
            report["gate_on" if gate else "gate_off"] = summarize(run_suite(suite, agents, gate=gate, jobs=1))
        report["reject_injection"] = {k: v["categories"].get("reject/injection") for k, v in report.items()}
    elif args.kaggle:
        agents, meter, pairs = kaggle_agents(args.kaggle)
        report = {"agents": pairs}
        for gate in ((True, False) if args.ablate else (True,)):
            before = meter.usd
            results = run_suite(suite, agents, gate=gate, jobs=args.jobs)
            summary = summarize(results)
            summary["cost_usd"] = round(meter.usd - before, 4)
            report["gate_on" if gate else "gate_off"] = summary
        report["calls"] = meter.calls
        report["failed_calls"] = meter.failures
    else:
        specs = {"chat": args.chat, "inbox": args.inbox, "web": args.web}
        if not any(specs.values()):
            ap.error("give --scripted, --kaggle, or at least one of --chat, --inbox, --web")
        agents = {name: provider_agent(spec) for name, spec in specs.items() if spec}
        report = summarize(run_suite(suite, agents, gate=True, jobs=args.jobs))
    report["seconds"] = round(time.time() - started)
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
