from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


# -----------------------------
# Time helpers
# -----------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(value: Any) -> Optional[datetime]:
    """
    Best-effort timestamp parsing.
    Supports:
    - datetime
    - ISO-8601 strings
    - unix seconds / milliseconds
    Returns None if parsing fails.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    if isinstance(value, (int, float)):
        # Heuristic: > 10^12 is probably ms
        try:
            if value > 1_000_000_000_000:
                return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except Exception:
            return None

    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    return None


# -----------------------------
# Key normalization
# -----------------------------

_KEY_RE = re.compile(r"[^a-z0-9]+")


def normalize_key(value: str) -> str:
    """
    Normalize keys to stable snake_case-ish lowercase tokens.
    """
    if not value:
        return ""
    value = value.strip().lower()
    value = _KEY_RE.sub("_", value)
    value = value.strip("_")
    return value


# -----------------------------
# Hashing / evidence
# -----------------------------

def sha256_json(obj: Any) -> str:
    """
    Produce a stable sha256 hash for JSON-serializable objects.
    """
    try:
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        payload = str(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# -----------------------------
# Safe stringify (for Interaction.text)
# -----------------------------

def safe_stringify(
    obj: Any,
    *,
    max_keys: int = 20,
    max_list: int = 20,
    max_str: int = 500,
) -> str:
    """
    Deterministically convert arbitrary objects to a compact human-readable string.

    Rules:
    - Prefer top-level scalar fields
    - Limit keys and list lengths
    - Truncate long strings
    - Avoid dumping entire blobs
    """
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[: max_str - 1] + "…"

    if isinstance(obj, (int, float, bool)):
        return str(obj)

    if isinstance(obj, list):
        items = obj[:max_list]
        rendered = [safe_stringify(v, max_str=max_str) for v in items]
        suffix = " …" if len(obj) > max_list else ""
        return f"[{', '.join(rendered)}]{suffix}"

    if isinstance(obj, dict):
        parts = []
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_keys:
                parts.append("…")
                break
            key = str(k)
            val = safe_stringify(v, max_str=max_str)
            parts.append(f"{key}: {val}")
        return "; ".join(parts)

    # Fallback
    return str(obj)


# -----------------------------
# Field picking helpers
# -----------------------------

def pick_first(obj: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    """
    Return the first non-empty value for the given keys.
    """
    for k in keys:
        if k in obj:
            v = obj.get(k)
            if v not in (None, "", [], {}):
                return v
    return None
