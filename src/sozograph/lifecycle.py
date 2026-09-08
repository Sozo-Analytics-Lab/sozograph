"""Conservative local erasure with source-group cascading."""
from __future__ import annotations

import json

from .evidence import entity_index, subjects


def references(row):
    data = row.to_compact()
    return set(data.get("source_ids", [])) | {data[k] for k in ("source", "source_old", "source_new") if data.get(k)} | {
        e["source"] for e in data.get("evidence", [])}


def forget(passport, *, source_ids=None, subject=None, contains=None):
    """Delete whole source groups to avoid residual evidence in summaries.

    This intentionally may remove unrelated details sharing a source segment.
    It cannot revoke external copies. Unknown extensions are cleared because
    their provenance semantics cannot be interpreted safely by this version.
    """
    passport.assert_supported()
    if not source_ids and not subject and not contains:
        raise ValueError("Specify source_ids, subject, or nonempty contains")
    names = entity_index(passport.entities, [subject] if subject else [])
    wanted = subjects(subject, names) if subject else set()
    target = set(source_ids or [])
    buckets = ("facts", "prefs", "observations", "episodes", "entities", "open_loops", "contradictions")

    def matches(row):
        text = json.dumps(row.to_compact(), ensure_ascii=False)
        return (bool(contains and contains.casefold() in text.casefold())
                or bool(wanted & subjects(text, names)))

    for source in passport.sources:
        if matches(source):
            target.add(source.id)
    for bucket in buckets:
        for row in getattr(passport, bucket):
            if matches(row):
                target.update(references(row))
    # If a derived record joins several sources, all of that record's source
    # groups are erased too; otherwise a surviving source could reveal it.
    changed = True
    while changed:
        before = len(target)
        for bucket in buckets:
            for row in getattr(passport, bucket):
                refs = references(row)
                if refs & target:
                    target.update(refs)
        changed = len(target) != before
    counts = {}
    for bucket in buckets:
        rows = getattr(passport, bucket)
        kept = [r for r in rows if not matches(r) and not references(r) & target]
        counts[bucket] = len(rows) - len(kept)
        setattr(passport, bucket, kept)
    kept_sources = [s for s in passport.sources if s.id not in target]
    counts["sources"] = len(passport.sources) - len(kept_sources)
    passport.sources = kept_sources
    # Audit/error payloads and extensions can contain the erased values.
    passport.meta.clear()
    if passport.model_extra:
        passport.model_extra.clear()
    passport.stats = []
    passport.ingest_report = {}
    for bucket in buckets:
        for row in getattr(passport, bucket):
            if row.model_extra:
                row.model_extra.clear()
    passport.touch()
    return counts
