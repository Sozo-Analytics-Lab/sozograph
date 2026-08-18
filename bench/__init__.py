"""
Benchmarks for SozoGraph.

Run from a checkout without installing anything:

    python -m bench.locomo.run --data data/locomo10.json --dry-run

The path guard below exists so that works. If `sozograph` is already installed,
this changes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    try:
        import sozograph  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(_SRC))
