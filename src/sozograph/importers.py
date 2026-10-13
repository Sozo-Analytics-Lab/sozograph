"""
Importers: memory from other assistants, as stamps with origin `imported`.

Vendors export memory as text (Claude's memory view, ChatGPT's saved memories,
Gemini), as files (CLAUDE.md and Claude Code memory notes), or as JSON (Letta
Agent File memory blocks, OpenMemory exports). Each importer turns one into
stamp proposals authored `import:<vendor>`. Lines become observations: plain,
attributable statements that recall ranks like any other. Pass an engine to
`import_text` when you want structured facts extracted from them as well.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schema import Passport

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)


def _lines(text: str) -> list[str]:
    out = []
    for raw in (text or "").splitlines():
        line = _BULLET.sub("", raw).strip()
        if not line or line.startswith("#") or len(line) < 3 or line.startswith("```"):
            continue
        out.append(line)
    return out


def _observations(lines: list[str], source: str, domains: list[str] | None = None) -> list[dict[str, Any]]:
    seen, items = set(), []
    for line in lines:
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        claim = {"text": line, "source": source}
        if domains:
            claim["domains"] = domains
        items.append({"op": "assert", "section": "observations", "claim": claim})
    return items


def from_text(text: str, *, vendor: str, domains: list[str] | None = None) -> list[dict[str, Any]]:
    """Any assistant's memory as text: one statement per line or bullet."""
    return _observations(_lines(text), f"import:{vendor}", domains)


def from_claude_memory(text: str) -> list[dict[str, Any]]:
    return from_text(text, vendor="claude")


def from_chatgpt_memory(text: str) -> list[dict[str, Any]]:
    return from_text(text, vendor="chatgpt")


def from_gemini(data: str | list | dict) -> list[dict[str, Any]]:
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return from_text(data, vendor="gemini")
    return _observations(_strings(data), "import:gemini")


def from_claude_md(text: str, *, domains: list[str] | None = None) -> list[dict[str, Any]]:
    """CLAUDE.md or a Claude Code memory note. Frontmatter names and descriptions are kept."""
    lines = []
    match = _FRONTMATTER.match(text or "")
    body = text or ""
    if match:
        body = body[match.end():]
        for row in match.group(1).splitlines():
            if row.strip().startswith(("description:", "name:")):
                lines.append(row.split(":", 1)[1].strip().strip('"'))
    lines += _lines(body)
    return _observations(lines, "import:claude-code", domains or ["software"])


def from_letta(data: str | dict) -> list[dict[str, Any]]:
    """Letta Agent File: the human memory blocks describe the user."""
    if isinstance(data, str):
        data = json.loads(data)
    blocks = []

    def walk(node):
        if isinstance(node, dict):
            if "label" in node and "value" in node and isinstance(node["value"], str):
                blocks.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    lines = [line for b in blocks if str(b.get("label", "")).lower() in {"human", "user", "persona_user"}
             for line in _lines(b["value"])]
    return _observations(lines, "import:letta")


def from_openmemory(data: str | list | dict) -> list[dict[str, Any]]:
    if isinstance(data, str):
        data = json.loads(data)
    return _observations(_strings(data), "import:openmemory")


def _strings(data: Any) -> list[str]:
    out: list[str] = []
    if isinstance(data, str):
        out.extend(_lines(data))
    elif isinstance(data, list):
        for item in data:
            out.extend(_strings(item))
    elif isinstance(data, dict):
        for key in ("memory", "content", "text", "value", "fact"):
            if isinstance(data.get(key), str):
                out.extend(_lines(data[key]))
                return out
        for key in ("memories", "items", "results", "data"):
            if key in data:
                out.extend(_strings(data[key]))
    return out


IMPORTERS = {"claude": from_claude_memory, "chatgpt": from_chatgpt_memory, "gemini": from_gemini,
             "claude-md": from_claude_md, "letta": from_letta, "openmemory": from_openmemory}


def import_into(passport: Passport, vendor: str, data: Any, *, now=None):
    """Propose an import as the holder's import author, so it lands with origin imported."""
    if vendor not in IMPORTERS:
        raise ValueError(f"Unknown importer {vendor}; choose from {', '.join(sorted(IMPORTERS))}")
    if isinstance(data, Path):
        data = data.read_text(encoding="utf-8")
    items = IMPORTERS[vendor](data)
    return passport.propose(items, author=f"import:{vendor}", now=now)
