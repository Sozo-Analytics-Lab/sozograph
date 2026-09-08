"""Local evidence alignment and entity identity; no model or index required."""
from __future__ import annotations

from .batching import Segment, source_id
from .retrieve import match_names, tokenize
from .schema import EvidenceRef, SourceRef


def entity_index(entities, names=()):
    groups: dict[str, set[str]] = {}
    for entity in entities:
        canonical = entity.name.casefold()
        groups.setdefault(canonical, set()).update([entity.name, *entity.aliases])
    for name in names:
        if not any(name.casefold() in {a.casefold() for a in aliases} for aliases in groups.values()):
            groups.setdefault(name.casefold(), set()).add(name)
    return groups


def subjects(text, index):
    return {name for name, aliases in index.items() if match_names(text, aliases)}


def align(text, segment: Segment):
    terms = set(tokenize(text))
    candidates = []
    for turn in segment.interactions:
        exact = turn.text.find(text)
        overlap = len(terms & set(tokenize(turn.text))) / max(1, len(terms))
        if exact >= 0 or overlap >= 0.45:
            candidates.append((exact >= 0, overlap, turn, exact))
    if not candidates:
        return None
    exact, score, turn, start = max(candidates, key=lambda x: (x[0], x[1]))
    quote = text if exact else turn.text[:500]
    # Candidate quotes are verbatim excerpts, but their support relationship
    # is heuristic. Never label overlap alignment as verified entailment.
    return EvidenceRef(source=source_id(segment), turn_id=str(turn.meta.get("parent_id") or turn.id),
                       quote=quote, start=start + int(turn.meta.get("span_start", 0)) if exact else None,
                       end=start + int(turn.meta.get("span_start", 0)) + len(text) if exact else None,
                       part_id=str(turn.id or ""),
                       match="exact" if exact else "candidate", score=1.0 if exact else score)


def attach(update, segment, *, retention="none", subject=""):
    entities = update.get("entities", [])
    names = entity_index(entities, segment.participants)
    for bucket in ("facts", "prefs", "observations", "episodes", "open_loops", "entities"):
        for record in update.get(bucket, []):
            text = (str(record.value) if bucket in {"facts", "prefs"}
                    else getattr(record, "text", getattr(record, "summary", getattr(record, "item", getattr(record, "name", "")))))
            ref = align(text, segment)
            if bucket == "entities":
                record.source_ids = [source_id(segment)]
            if ref and retention != "none":
                record.evidence = [ref]
            if hasattr(record, "subject"):
                explicit = subjects(record.search_text(), names)
                record.subject = subject or (next(iter(explicit)) if len(explicit) == 1 else "")
                if not record.subject and ref:
                    turn = next((i for i in segment.interactions
                                 if str(i.meta.get("parent_id") or i.id) == ref.turn_id), None)
                    if turn and set(tokenize(turn.text, keep_stopwords=True)) & {"i", "my"}:
                        record.subject = str(turn.meta.get("speaker") or "")
            if hasattr(record, "participants"):
                found = subjects(text, names)
                record.participants = sorted(next((a for a in names[n] if a.casefold() == n), n) for n in found)


def source_record(segment: Segment, *, retention="none", update=None):
    from .batching import interaction_identity
    from .utils import sha256_json

    turns = [{"id": str(i.meta.get("parent_id") or i.id), "part_id": i.id,
              "text": i.text, "speaker": i.meta.get("speaker", ""),
              "ts": None if i.meta.get("timestamp_missing") else i.ts.isoformat(),
              "start": i.meta.get("span_start", 0)} for i in segment.interactions]
    kept = turns if retention == "full" else []
    if retention == "excerpts" and update:
        seen = set()
        for bucket in ("facts", "prefs", "observations", "episodes", "open_loops", "entities"):
            for record in update.get(bucket, []):
                for ref in record.evidence:
                    if (ref.turn_id, ref.quote) not in seen:
                        kept.append({"id": ref.turn_id, "text": ref.quote, "match": ref.match})
                        seen.add((ref.turn_id, ref.quote))
    kinds = {"transcript", "firestore", "rtdb", "supabase", "chat", "form", "unknown"}
    return SourceRef(id=source_id(segment), kind=segment.type if segment.type in kinds else "unknown",
                     ts=segment.ts, source=segment.source,
                     hash=sha256_json([interaction_identity(i) for i in segment.interactions]),
                     turn_ids=list(dict.fromkeys(t["id"] for t in turns)), turns=kept,
                     retention=retention, coverage="full" if retention == "full" else "partial" if kept else "none")
