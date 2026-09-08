"""Repeatable offline recall latency and storage profile; no QA accuracy claim."""
from __future__ import annotations

import argparse
import math
import platform
import statistics
import time
from datetime import datetime, timezone

from sozograph import Entity, Observation, Passport, __version__

from .replay import code_hash, write_json


def profile(*, sizes=(100, 1000, 10000), repeats=7):
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for count in sizes:
        p = Passport(updated_at=stamp, entities=[Entity(name=f"Person{i}") for i in range(10)],
                     observations=[Observation(text=f"Person{i % 10} visited museum {i} and bought a blue notebook.",
                                               when=f"2026-01-{i % 28 + 1:02d}", ts=stamp, source=f"s{i}")
                                   for i in range(count)])
        times, lengths = [], []
        for repeat in range(repeats + 1):
            started = time.perf_counter()
            result = p.recall(query="Which museums did Person3 visit?", budget_chars=6000)
            elapsed = (time.perf_counter() - started) * 1000
            if repeat:
                times.append(elapsed)
                lengths.append(result.used_chars)
        rows.append({"records": count, "p50_ms": statistics.median(times),
                     "p95_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1],
                     "samples_ms": times, "storage_bytes": len(p.to_json(indent=None).encode("utf-8")),
                     "context_chars": max(lengths), "budget_chars": 6000})
    return {"version": __version__, "code_hash": code_hash(), "python": platform.python_version(),
            "platform": platform.platform(), "repeats": repeats, "warmups": 1,
            "note": "Synthetic offline timing; nearest-rank p95; no provider calls or accuracy evaluation", "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="bench/results/offline_profile.json")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    result = profile(repeats=args.repeats)
    write_json(args.out, result)
    print(args.out)


if __name__ == "__main__":
    main()
