"""Offline structural integrity checks for a portable passport."""
from __future__ import annotations

from collections import Counter

from .lifecycle import references


def audit(passport):
    passport.assert_supported()
    issues = []
    counts = Counter(s.id for s in passport.sources)
    for sid, count in counts.items():
        if count > 1:
            issues.append({"code": "duplicate_source", "source": sid, "count": count})
    sources = {s.id: s for s in passport.sources}
    for bucket in ("facts", "prefs", "observations", "episodes", "entities", "open_loops", "contradictions"):
        for index, record in enumerate(getattr(passport, bucket)):
            location = f"{bucket}/{index}"
            for sid in references(record) - set(sources):
                issues.append({"code": "missing_source", "record": location, "source": sid})
            for ref in record.evidence:
                if ref.match != "exact":
                    continue
                if ref.start is None or ref.end is None or ref.start < 0 or ref.end - ref.start != len(ref.quote):
                    issues.append({"code": "invalid_span", "record": location})
                    continue
                source = sources.get(ref.source)
                if not source or source.coverage != "full":
                    continue
                matches = [t for t in source.turns if str(t.get("part_id")) == ref.part_id]
                if not matches:
                    issues.append({"code": "missing_evidence_turn", "record": location, "turn": ref.turn_id})
                elif not any(t["text"][ref.start - int(t.get("start", 0)):ref.end - int(t.get("start", 0))] == ref.quote
                             for t in matches):
                    issues.append({"code": "span_mismatch", "record": location})
    return {"valid": not issues, "issues": issues, "sources": len(sources),
            "storage_bytes": len(passport.to_json(indent=None).encode("utf-8")),
            "content_hash": passport.content_hash()}
