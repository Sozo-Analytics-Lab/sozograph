"""Atomic local benchmark artifacts and matched outcome statistics."""
from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

from sozograph.utils import sha256_json


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as f:
            name = f.name
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def code_hash():
    root = Path(__file__).resolve().parents[1]
    files = sorted([*root.joinpath("src").rglob("*.py"), *root.joinpath("bench").rglob("*.py")])
    return sha256_json({str(p.relative_to(root)): p.read_text(encoding="utf-8") for p in files})


def coverage(gold, selected):
    gold = set(gold)
    return len(gold & set(selected)) / len(gold) if gold else None


def paired_bootstrap(left, right, *, iterations=2000, seed=0):
    """Question-weighted accuracy difference, resampled by conversation."""
    if set(left) != set(right) or not left:
        raise ValueError("Paired runs must contain identical nonempty conversation sets")
    for key in left:
        if len(left[key]) != len(right[key]) or not left[key]:
            raise ValueError("Paired conversations must contain identical question counts")
    if iterations < 40:
        raise ValueError("Use at least 40 bootstrap draws")
    rng = random.Random(seed)
    keys = sorted(left)

    def delta(sample):
        return sum(sum(right[k]) - sum(left[k]) for k in sample) / sum(len(left[k]) for k in sample)
    draws = sorted(delta(rng.choices(keys, k=len(keys))) for _ in range(iterations))
    return {"difference": delta(keys), "low": draws[int(iterations * 0.025)],
            "high": draws[min(iterations - 1, int(iterations * 0.975))],
            "clusters": len(keys), "iterations": iterations, "seed": seed}
