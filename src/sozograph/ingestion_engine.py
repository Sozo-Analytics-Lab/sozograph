"""Bounded, resumable ingestion with explicit completeness diagnostics."""
from __future__ import annotations

from .batching import Segment, segment_interactions, source_id, split_interaction
from .evidence import attach, source_record
from .extractor import Extractor
from .ingest import apply_fallback_summaries, coerce_to_interactions
from .resolver import merge_passport_update
from .schema import Passport, utcnow
from .utils import sha256_json, stable_id


def ingest_into(engine, data, *, passport, meta, hint, batch, max_segment_tokens,
                retention, extraction_revision, reextract, max_extra_calls,
                max_split_depth, clock, checkpoint):
    if retention not in {"none", "excerpts", "full"}:
        raise ValueError("retention must be none, excerpts, or full")
    if max_extra_calls < 0 or max_split_depth < 0:
        raise ValueError("Retry bounds must be nonnegative")
    now = clock or utcnow
    base = passport if passport is not None else Passport(updated_at=now())
    base.assert_supported()
    meta = dict(meta or {})
    if clock and "ts" not in meta:
        meta["ts"] = now().isoformat()
    if meta.get("user_key"):
        if base.user_key and base.user_key != str(meta["user_key"]):
            raise ValueError("Cannot ingest a different user into this passport")
        base.user_key = str(meta["user_key"])
    interactions, sources = coerce_to_interactions(data, hint=hint, meta=meta)
    normalize_database_text(interactions)
    if not interactions:
        base.ingest_report = {"complete": True, "calls": 0, "units": {}}
        return base
    if batch:
        units = segment_interactions(interactions, max_tokens=max_segment_tokens)
    else:
        units = [s for it in interactions for s in segment_interactions([it], max_tokens=max_segment_tokens)]
    import os
    spec = engine._provider_spec
    provider_spec = (spec if isinstance(spec, str) else f"{spec.name}:{spec.model}" if spec
                     else "environment:" + os.getenv("SOZOGRAPH_PROVIDER", "auto") + ":"
                     + os.getenv("SOZOGRAPH_MODEL", "default"))
    config = {"revision": extraction_revision, "provider": provider_spec,
              "options": {k: v for k, v in engine._provider_kwargs.items() if k != "api_key"},
              "retention": retention, "subject": meta.get("subject", ""),
              "segment_tokens": max_segment_tokens, "batch": batch}
    revision = sha256_json(config)
    ledger = base.meta.setdefault("ingest", {}).setdefault(revision, {})
    report = {"complete": False, "calls": 0, "skipped": 0, "rejected_records": 0,
              "saturated_units": 0, "units": {}, "revision": extraction_revision,
              "retention": retention}
    stats = []
    allowance = len(units) + max_extra_calls

    def checkpoint_now():
        base.ingest_report = report
        base.stats = stats
        if checkpoint:
            checkpoint(base)

    def run(segment, depth=0):
        sid = source_id(segment)
        key = sid
        if not reextract and (ledger.get(key) == "complete" or isinstance(ledger.get(key), dict) and ledger[key].get("status") == "complete"):
            report["skipped"] += 1
            report["units"][sid] = {"status": "skipped"}
            return True
        if report["calls"] >= allowance:
            report["units"][sid] = {"status": "pending", "reason": "call budget exhausted"}
            return False
        report["calls"] += 1
        extractor = Extractor(engine.provider)
        update, error = None, None
        try:
            update = extractor.extract_segment(segment, known_keys=base.known_keys())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
        diagnostics = getattr(extractor, "diagnostics", {})
        rejected = diagnostics.get("rejected_records", 0)
        saturated = bool(diagnostics.get("saturated"))
        report["rejected_records"] += rejected
        report["saturated_units"] += int(saturated)
        needs_retry = bool(error or rejected or saturated)
        # Keep useful output even when a coverage diagnostic requests repair.
        if update is not None:
            attach(update, segment, retention=retention, subject=str(meta.get("subject") or ""))
            if reextract and not needs_retry:
                for bucket in ("facts", "prefs", "observations", "episodes", "open_loops"):
                    rows = getattr(base, bucket)
                    rows[:] = [r for r in rows if r.source != sid or set(r.source_ids) - {sid}]
            base.upsert_source(source_record(segment, retention=retention, update=update))
            _, result = merge_passport_update(base, now=now(), **update)
            stats.append(result)
        else:
            base.upsert_source(source_record(segment, retention=retention))
        ok = not needs_retry
        children = []
        if needs_retry and depth < max_split_depth and report["calls"] < allowance:
            turns = segment.interactions
            if len(turns) > 1:
                middle = len(turns) // 2
                groups = [turns[:middle], turns[middle:]]
            else:
                try:
                    parts = split_interaction(turns[0], max(200, len(segment.text()) // 2))
                except ValueError:
                    parts = turns
                groups = [[part] for part in parts] if len(parts) > 1 else []
            for index, group in enumerate(groups):
                children.append(Segment(id=stable_id("repair_", [segment.id, depth, index], length=32),
                                        interactions=group, boundary_reason="coverage repair"))
            if children:
                outcomes = [run(child, depth + 1) for child in children]
                ok = all(outcomes)
        status = "complete" if ok else "partial" if update is not None else "failed"
        details = {"status": status}
        if error:
            details["error"] = error
            failures = base.meta.setdefault("ingest_failures", [])
            failure = {"segment": segment.id, "error": error}
            if failure not in failures:
                failures.append(failure)
        if rejected:
            # A handful of reasons, not the count again: what the model
            # produced that the wire schema (or a pydantic-only constraint
            # the model was never told about, e.g. a required min_length)
            # rejected is exactly what turns "why is this incomplete" from
            # a guess into a fix.
            details["rejected_reasons"] = diagnostics.get("rejected_reasons", [])[:3]
        if children:
            details["children"] = [source_id(child) for child in children]
        ledger[key] = "complete" if ok and not children else details
        report["units"][sid] = details
        checkpoint_now()
        return ok

    # Normalized database data is readable JSON. Fallback remains available
    # only for genuinely weak adapter output; no human turn is summarized.
    if any(it.type == "unknown" and not it.data for it in interactions):
        apply_fallback_summaries(interactions, sources=sources, provider=engine.provider, cfg=engine.ingest_cfg)
    outcomes = [run(unit) for unit in units]
    report["complete"] = all(outcomes)
    report["storage_bytes"] = len(base.to_json(indent=None).encode("utf-8"))
    checkpoint_now()
    return base


def normalize_database_text(interactions):
    """Expand adapter projections from their available original data."""
    import json
    for it in interactions:
        if it.type not in {"chat", "transcript", "note", "email", "message"} and it.data:
            it.text = json.dumps(it.data, ensure_ascii=False, default=str)
