from __future__ import annotations

from typing import Any, List, Tuple

from .schema import Passport, Fact, Preference, Entity, OpenLoop, Contradiction
from .utils import normalize_key


def _val_to_str(v: Any, max_len: int = 220) -> str:
    if v is None:
        s = "null"
    elif isinstance(v, bool):
        s = "true" if v else "false"
    elif isinstance(v, (int, float)):
        s = str(v)
    elif isinstance(v, str):
        s = v.strip()
    else:
        # compact repr for simple JSON-ish values
        s = str(v)

    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _score_item(ts, confidence: float) -> float:
    # Simple v1 scoring: prefer newer + higher confidence
    # (No explicit decay weights in v1)
    try:
        t = ts.timestamp()
    except Exception:
        t = 0.0
    return (t / 1_000_000_000.0) + (confidence * 0.5)


def _pick_top_facts(facts: List[Fact], n: int) -> List[Fact]:
    ranked = sorted(facts, key=lambda f: _score_item(f.ts, float(f.confidence)), reverse=True)
    return ranked[:n]


def _pick_top_prefs(prefs: List[Preference], n: int) -> List[Preference]:
    ranked = sorted(prefs, key=lambda p: _score_item(p.ts, float(p.confidence)), reverse=True)
    return ranked[:n]


def _pick_top_open_loops(open_loops: List[OpenLoop], n: int) -> List[OpenLoop]:
    ranked = sorted(open_loops, key=lambda o: o.ts, reverse=True)
    return ranked[:n]


def _pick_top_contradictions(contradictions: List[Contradiction], n: int) -> List[Contradiction]:
    ranked = sorted(contradictions, key=lambda c: c.ts_new, reverse=True)
    return ranked[:n]


def _entities_summary(entities: List[Entity], max_items: int = 12) -> List[str]:
    # Keep it short; entities are optional context, not the main payload
    out: List[str] = []
    for e in entities[:max_items]:
        if e.type and e.type != "other":
            out.append(f"{e.name} ({e.type})")
        else:
            out.append(e.name)
    return out


def export_context(
    passport: Passport,
    *,
    budget_chars: int = 3000,
    header: str = "SOZOGRAPH PASSPORT v1",
) -> str:
    """
    Render a compact, stable context string.

    Strategy:
    - Start with a stable header
    - Include: facts, prefs, entities, open loops, contradictions (in that order)
    - Respect budget by trimming least-important sections first
    """
    # Guard rails
    budget_chars = max(400, int(budget_chars or 3000))

    # Pick reasonable caps (we'll trim further if needed)
    facts = _pick_top_facts(passport.facts, n=25)
    prefs = _pick_top_prefs(passport.prefs, n=15)
    open_loops = _pick_top_open_loops(passport.open_loops, n=10)
    contradictions = _pick_top_contradictions(passport.contradictions, n=8)
    entities = passport.entities or []

    lines: List[str] = []
    lines.append(header)
    if passport.user_key:
        lines.append(f"User: {passport.user_key}")
    lines.append(f"Updated: {passport.updated_at.isoformat()}")

    # Facts
    if facts:
        lines.append("")
        lines.append("Facts (current beliefs):")
        for f in facts:
            lines.append(f"- {normalize_key(f.key)}: {_val_to_str(f.value)}")

    # Preferences
    if prefs:
        lines.append("")
        lines.append("Preferences:")
        for p in prefs:
            lines.append(f"- {normalize_key(p.key)}: {_val_to_str(p.value)}")

    # Entities
    ent_lines = _entities_summary(entities)
    if ent_lines:
        lines.append("")
        lines.append("Key entities:")
        for s in ent_lines:
            lines.append(f"- {s}")

    # Open loops
    if open_loops:
        lines.append("")
        lines.append("Open loops:")
        for o in open_loops:
            lines.append(f"- {_val_to_str(o.item, max_len=240)}")

    # Contradictions
    if contradictions:
        lines.append("")
        lines.append("Recent updates (contradictions resolved by time):")
        for c in contradictions:
            lines.append(
                f"- {normalize_key(c.key)} changed: {_val_to_str(c.old)} -> {_val_to_str(c.new)}"
            )

    # Budget enforcement (trim bottom-up)
    def join_len(ls: List[str]) -> int:
        return len("\n".join(ls))

    # If over budget, progressively trim sections by reducing item counts
    if join_len(lines) > budget_chars:
        # Helper to rebuild with smaller caps
        def rebuild(f_n: int, p_n: int, e_n: int, o_n: int, c_n: int) -> List[str]:
            f2 = _pick_top_facts(passport.facts, n=f_n)
            p2 = _pick_top_prefs(passport.prefs, n=p_n)
            o2 = _pick_top_open_loops(passport.open_loops, n=o_n)
            c2 = _pick_top_contradictions(passport.contradictions, n=c_n)
            e2 = passport.entities[:e_n] if passport.entities else []

            out: List[str] = []
            out.append(header)
            if passport.user_key:
                out.append(f"User: {passport.user_key}")
            out.append(f"Updated: {passport.updated_at.isoformat()}")

            if f2:
                out.append("")
                out.append("Facts (current beliefs):")
                for f in f2:
                    out.append(f"- {normalize_key(f.key)}: {_val_to_str(f.value)}")

            if p2:
                out.append("")
                out.append("Preferences:")
                for p in p2:
                    out.append(f"- {normalize_key(p.key)}: {_val_to_str(p.value)}")

            ent2 = _entities_summary(e2, max_items=e_n)
            if ent2:
                out.append("")
                out.append("Key entities:")
                for s in ent2:
                    out.append(f"- {s}")

            if o2:
                out.append("")
                out.append("Open loops:")
                for o in o2:
                    out.append(f"- {_val_to_str(o.item, max_len=240)}")

            if c2:
                out.append("")
                out.append("Recent updates (contradictions resolved by time):")
                for c in c2:
                    out.append(
                        f"- {normalize_key(c.key)} changed: {_val_to_str(c.old)} -> {_val_to_str(c.new)}"
                    )

            return out

        # Start trimming least important first: contradictions, open loops, entities, prefs, facts
        caps = [25, 15, 12, 10, 8]  # f, p, e, o, c
        for _ in range(80):
            if join_len(lines) <= budget_chars:
                break

            # pick a trim step
            f_n, p_n, e_n, o_n, c_n = caps
            if c_n > 0:
                c_n = max(0, c_n - 1)
            elif o_n > 0:
                o_n = max(0, o_n - 1)
            elif e_n > 0:
                e_n = max(0, e_n - 1)
            elif p_n > 0:
                p_n = max(0, p_n - 1)
            elif f_n > 5:
                f_n = max(5, f_n - 1)
            else:
                # last resort: hard truncate joined text
                txt = "\n".join(lines)
                return txt[: budget_chars - 1] + "…"

            caps = [f_n, p_n, e_n, o_n, c_n]
            lines = rebuild(f_n, p_n, e_n, o_n, c_n)

    return "\n".join(lines)
