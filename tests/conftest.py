from __future__ import annotations

import sys
from pathlib import Path

# The package lives in src/ and is not necessarily installed. Without this a
# bare `pytest` fails at collection with import errors.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
