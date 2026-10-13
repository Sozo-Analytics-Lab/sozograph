"""
The stamp protocol: how agents and the holder write to a log-backed passport.

    result = passport.propose([...], visa=muse)    # agents write stamps
    passport.pending(); passport.confirm(ids)      # the holder decides quarantine
    merged = Passport.merge(laptop, hosted)        # replicas converge by union

The host agent supplies the claim, the operation and optionally the stamp it
replaces. Everything else (identity, author, clock, defaults, key
canonicalization, admission) is filled here, so a writer only does the part
that needs judgment.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from .dedupe import find_match
from .gate import REJECTED, admit, scan_secret
from .projection import MODELS, analyse, claim_time, project
from .schema import (
    ORIGIN_RANK,
    STAMP_SECTIONS,
    AgentRecord,
    Passport,
    Stamp,
    Visa,
    utcnow,
)
from .shelf import infer_kind, parse_moment
from .stamps import Clock, canonical, hlc_encode, merkle_root, redact, stamp_id, union
from .utils import normalize_key, sha256_json

HOLDER = "holder"
AGENT_ORIGINS = {"user_direct", "agent_inferred", "tool_result", "third_party"}
_ITEM_KEYS = {"op", "section", "claim", "targets", "target", "origin", "evidence", "reason"}
_HOLDER_WORDS = {"user", "holder", "me", "i", "self", "the user", "the holder"}


@dataclass
class ProposeResult:
    accepted: list[dict[str, Any]] = field(default_factory=list)
    quarantined: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def ids(self) -> list[str]:
        return [r["id"] for r in self.accepted + self.quarantined]


# -- genesis ---------------------------------------------------------------

def ensure_log(passport: Passport) -> None:
    """Turn a snapshot into genesis stamps, once, so the log becomes canonical."""
    passport.assert_supported()
    if not passport.holder.get("id"):
        passport.holder = {**passport.holder, "id": "holder"}
    if passport.stamps:
        return
    counter = 0
    seen: set[str] = set()
    for section in STAMP_SECTIONS:
        for record in getattr(passport, section):
            claim = record.to_compact()
            claim.pop("stamp_ids", None)
            moment = getattr(record, "ts", None) or getattr(record, "ts_new", None)
            wall = int(moment.timestamp() * 1000) if moment else 0
            stamp = Stamp(op="assert", section=section, claim=claim, author="import:passport",
                          origin="imported", hlc=hlc_encode(max(wall, 0), counter, "genesis"))
            counter += 1
            stamp.id = stamp_id(stamp)
            if stamp.id in seen:
                continue
            seen.add(stamp.id)
            passport.stamps.append(stamp)
    if passport.stamps:
        project(passport)


# -- normalization ------------------------------------------------------------

def _section_of(claim: dict[str, Any]) -> str:
    if "text" in claim:
        return "observations"
    if "item" in claim:
        return "open_loops"
    if "summary" in claim:
        return "episodes"
    if "name" in claim and "value" not in claim:
        return "entities"
    return "facts"


def _current_ids(passport: Passport, section: str, claim: dict[str, Any]) -> list[str]:
    if section not in {"facts", "prefs", "open_loops"}:
        return []
    subject = str(claim.get("subject", "")).casefold()
    if section == "open_loops":
        from .resolver import _loop_key
        want = _loop_key(claim.get("item", ""))
        rows = [r for r in passport.open_loops if r.subject.casefold() == subject and _loop_key(r.item) == want]
    else:
        want = normalize_key(str(claim.get("key", "")))
        rows = [r for r in getattr(passport, section)
                if r.subject.casefold() == subject and normalize_key(r.key) == want]
    return sorted({sid for r in rows for sid in r.stamp_ids})


def _normalize(passport: Passport, item: Any, *, writer: str, visa: Visa | None, origin: str | None,
               moment: datetime, batch_keys: dict) -> Stamp:
    if isinstance(item, Stamp):
        item = item.model_dump(mode="json", exclude={"id", "admission", "admission_reason", "sig", "redacted"})
    if not isinstance(item, dict):
        raise ValueError("a stamp is an object")
    op = item.get("op", "assert")
    claim = dict(item["claim"]) if isinstance(item.get("claim"), dict) else \
        {k: v for k, v in item.items() if k not in _ITEM_KEYS}
    claim.pop("stamp_ids", None)
    claim.pop("section", None)
    if "attribute" in claim and "key" not in claim:
        claim["key"] = claim.pop("attribute")
    if str(claim.get("subject", "")).strip().casefold() in _HOLDER_WORDS:
        claim["subject"] = ""
    section = item.get("section") or (claim and _section_of(claim)) or ""
    if section and section not in MODELS:
        raise ValueError(f"unknown section {section}")
    if section == "contradictions" and writer != HOLDER and not writer.startswith("import:"):
        raise ValueError("contradictions are derived; write a supersede instead")
    targets = list(item.get("targets") or ([item["target"]] if item.get("target") else []))

    requested = item.get("origin") or origin
    if writer == HOLDER:
        stamp_origin = requested or ("user_confirmed" if op == "confirm" else "user_direct")
    elif writer.startswith("import:"):
        stamp_origin = "imported"
    else:
        stamp_origin = requested or "agent_inferred"
        if stamp_origin == "user_confirmed":
            stamp_origin = "user_direct"
        if stamp_origin not in AGENT_ORIGINS:
            stamp_origin = "agent_inferred"

    if op in {"confirm", "forget"}:
        claim = {}
        if not targets:
            raise ValueError(f"{op} needs targets")
    if op == "retract" and not claim.get("valid_until"):
        claim = {k: v for k, v in claim.items() if k in {"subject", "key", "item"}} if not targets else {}

    if claim and op in {"assert", "supersede", "correct"}:
        _fill_defaults(passport, section, claim, writer=writer, visa=visa, moment=moment, batch_keys=batch_keys)
    if op in {"supersede", "correct", "retract", "expire"} and not targets:
        targets = _current_ids(passport, section, claim)
        if not targets:
            if op in {"supersede", "correct"}:
                op = "assert"
            else:
                raise ValueError(f"nothing current to {op}")
    if op in {"retract", "expire"}:
        claim = {"valid_until": str(claim["valid_until"])} if claim.get("valid_until") else {}
        if op == "expire" and not claim:
            raise ValueError("expire needs claim.valid_until")
    if op in {"assert", "supersede"} and not claim:
        raise ValueError(f"{op} needs a claim")
    evidence = [e for e in (item.get("evidence") or []) if isinstance(e, dict)]
    evidence = json.loads(canonical(evidence)) if evidence else []
    stamp = Stamp(op=op, section=section if claim or op not in {"confirm", "forget"} else "",
                  claim=claim, targets=sorted(set(targets)), author=writer,
                  visa=visa.id if visa else "", origin=stamp_origin, evidence=evidence,
                  reason=str(item.get("reason", ""))[:200])
    if stamp.claim and stamp.section and op in {"assert", "supersede", "correct"}:
        _validate(stamp)
    return stamp


def _fill_defaults(passport, section, claim, *, writer, visa, moment, batch_keys):
    if section in {"facts", "prefs", "observations", "open_loops", "episodes"}:
        claim.setdefault("source", writer)
    if section in {"facts", "prefs", "observations", "open_loops", "episodes"} and "ts" not in claim:
        claim["ts"] = moment.isoformat()
    if section == "episodes" and "id" not in claim:
        claim["id"] = "ep_" + sha256_json([claim.get("summary"), claim.get("ts")])[:24]
    for key in ("valid_from", "valid_until"):
        if claim.get(key):
            if parse_moment(claim[key]) is None:
                raise ValueError(f"{key} must be an ISO date")
            claim[key] = str(claim[key])
    if visa is not None and not claim.get("domains"):
        home = [d for d in visa.domains if d != "*"]
        claim["domains"] = [home[0]] if home else ["general"]
    if claim.get("domains"):
        claim["domains"] = sorted({str(d).strip().lower() for d in claim["domains"] if str(d).strip()})
    if section in {"facts", "prefs"}:
        if not claim.get("key"):
            raise ValueError("facts and preferences need a key")
        key = normalize_key(str(claim["key"]))
        subject = str(claim.get("subject", "")).casefold()
        rows = getattr(passport, section)
        known = sorted({normalize_key(r.key) for r in rows if r.subject.casefold() == subject}
                       | set(batch_keys.get((section, subject), set())))
        match = find_match(key, known)
        if match.is_merge:
            key = normalize_key(match.existing)
        claim["key"] = key
        batch_keys.setdefault((section, subject), set()).add(key)
        if not claim.get("kind") and not writer.startswith(("import:", "sozograph:")):
            claim["kind"] = "preference" if section == "prefs" else infer_kind(key, claim.get("value", ""))


def _validate(stamp: Stamp) -> None:
    data = dict(stamp.claim)
    try:
        MODELS[stamp.section].model_validate(data)
    except Exception as exc:  # pydantic ValidationError, kept short for the agent
        first = str(exc).splitlines()
        raise ValueError("invalid claim: " + " ".join(first[:3])[:300]) from None


def _summary(stamp: Stamp) -> str:
    c = stamp.claim
    if stamp.section in {"facts", "prefs"}:
        scope = f"{c.get('subject')}: " if c.get("subject") else ""
        return f"{scope}{c.get('key')}: {json.dumps(c.get('value'), ensure_ascii=False)[:120]}"
    for key in ("text", "item", "summary", "name"):
        if c.get(key):
            return str(c[key])[:160]
    return f"{stamp.op} {','.join(stamp.targets)[:120]}"


# -- writing -----------------------------------------------------------------

def _clock_for(passport: Passport, moment: datetime, node: str) -> tuple[Clock, str]:
    clock = Clock(node, now=lambda: moment)
    latest = max((s.hlc for s in passport.stamps), default="")
    clock.observe(latest)
    return clock, latest


def _register(passport: Passport, writer: str, visa: Visa | None, moment: datetime) -> None:
    if writer == HOLDER or writer.startswith("import:"):
        return
    record = passport.agents.get(writer) or AgentRecord(id=writer, first_seen=moment)
    if visa is not None and not record.home_domains:
        record.home_domains = [d for d in visa.domains if d != "*"]
    record.last_seen = moment
    passport.agents[writer] = record


def propose(passport: Passport, items: list[Any], *, visa: Visa | None = None, author: str | None = None,
            origin: str | None = None, now: datetime | None = None, node: str = "local") -> ProposeResult:
    """Validate, gate and append stamps, then reproject. Rejected stamps never touch the file."""
    passport.assert_supported()
    ensure_log(passport)
    moment = now or utcnow()
    writer = visa.agent if visa is not None else (author or HOLDER)
    if visa is not None and not visa.active(moment):
        raise PermissionError("visa is inactive")
    clock, latest = _clock_for(passport, moment, node)
    ids = {s.id for s in passport.stamps}
    result = ProposeResult()
    batch_keys: dict = {}
    for index, item in enumerate(items if isinstance(items, list) else [items]):
        try:
            stamp = _normalize(passport, item, writer=writer, visa=visa, origin=origin,
                               moment=moment, batch_keys=batch_keys)
        except ValueError as exc:
            result.rejected.append({"index": index, "reason": str(exc)})
            continue
        stamp.hlc = clock.tick()
        stamp.seen = latest
        verdict, reason = admit(passport, stamp, visa, now=moment)
        if verdict == REJECTED:
            result.rejected.append({"index": index, "reason": reason})
            continue
        stamp.id = stamp_id(stamp)
        if stamp.id in ids:
            result.duplicates.append(stamp.id)
            continue
        stamp.admission, stamp.admission_reason = verdict, reason
        if passport._signer is not None:
            stamp.sig = passport._signer.sign(stamp)
        passport.stamps.append(stamp)
        ids.add(stamp.id)
        if stamp.op == "forget" and verdict == "accepted":
            _erase(passport, stamp)
        row = {"id": stamp.id, "op": stamp.op, "section": stamp.section, "summary": _summary(stamp)}
        if verdict == "quarantined":
            result.quarantined.append({**row, "reason": reason})
        else:
            result.accepted.append(row)
    if result.accepted or result.quarantined:
        _register(passport, writer, visa, moment)
        project(passport)
        passport.touch(moment)
    return result


def _erase(passport: Passport, forget: Stamp) -> None:
    targets = set(forget.targets)
    passport.stamps = [redact(s) if s.id in targets and not s.redacted else s for s in passport.stamps]
    erased = set(forget.erase_sources)
    if erased:
        passport.sources = [s for s in passport.sources if s.id not in erased]
    for key in ("dedupe", "ingest_failures", "projection"):
        passport.meta.pop(key, None)


def confirm(passport: Passport, ids: list[str], *, now=None) -> ProposeResult:
    return propose(passport, [{"op": "confirm", "targets": [i]} for i in ids], author=HOLDER, now=now)


def dismiss(passport: Passport, ids: list[str], *, now=None) -> ProposeResult:
    """Holder says these stamps were false from the start."""
    return propose(passport, [{"op": "correct", "targets": [i], "claim": {}} for i in ids],
                   author=HOLDER, now=now) if ids else ProposeResult()


def pending(passport: Passport) -> list[dict[str, Any]]:
    if not passport.stamps:
        return []
    state = analyse(passport.stamps)
    done = state.confirmed | state.corrected | state.forgotten | state.retracted
    rows = []
    for s in passport.stamps:
        if s.admission == "quarantined" and s.id not in done and not s.redacted:
            rows.append({"id": s.id, "author": s.author, "op": s.op, "section": s.section,
                         "summary": _summary(s), "reason": s.admission_reason, "origin": s.origin,
                         "hlc": s.hlc})
    return rows


# -- forgetting ------------------------------------------------------------------

def _refs(stamp: Stamp) -> set[str]:
    c = stamp.claim
    refs = set(c.get("source_ids") or [])
    for key in ("source", "source_old", "source_new"):
        if c.get(key):
            refs.add(str(c[key]))
    for e in c.get("evidence") or []:
        if isinstance(e, dict) and e.get("source"):
            refs.add(str(e["source"]))
    return refs


def forget(passport: Passport, *, source_ids=None, subject=None, contains=None, stamp_ids=None,
           visa: Visa | None = None, now=None, visible=None) -> dict[str, int]:
    """
    Erase claims everywhere this passport's log reaches.

    `visible` limits matching and cascading to stamps an agent may see, so a
    visa with forget permission erases inside its own domains only.

    Matching stamps are redacted in place and named by one forget stamp, so the
    erasure survives every later merge with a stale copy. Like 2.x forgetting,
    whole source groups go with them, which may remove unrelated details that
    shared a source segment.
    """
    from .evidence import entity_index, subjects

    ensure_log(passport)
    if not (source_ids or subject or contains or stamp_ids):
        raise ValueError("Specify stamp_ids, source_ids, subject, or nonempty contains")
    names = entity_index(passport.entities, [subject] if subject else [])
    wanted = subjects(subject, names) if subject else set()
    needle = (contains or "").casefold()
    sources = {s.id: s for s in passport.sources}
    target_sources = set(source_ids or [])
    targets = set(stamp_ids or [])
    if visible is not None:
        by_id = {s.id: s for s in passport.stamps}
        targets = {t for t in targets if t in by_id and visible(by_id[t])}

    def matches(text: str) -> bool:
        return bool((needle and needle in text.casefold()) or (wanted and wanted & subjects(text, names)))

    if visible is None:
        for sid, source in sources.items():
            if matches(json.dumps(source.to_compact(), ensure_ascii=False)):
                target_sources.add(sid)
    for s in passport.stamps:
        if s.redacted or not s.claim or (visible is not None and not visible(s)):
            continue
        body = json.dumps({"claim": s.claim, "evidence": s.evidence}, ensure_ascii=False)
        if matches(body) or _refs(s) & target_sources:
            targets.add(s.id)
    changed = True
    while changed:
        before = (len(targets), len(target_sources))
        for s in passport.stamps:
            if s.id in targets:
                target_sources |= _refs(s) & set(sources)
            elif (s.claim and not s.redacted and _refs(s) & target_sources
                  and (visible is None or visible(s))):
                targets.add(s.id)
        changed = (len(targets), len(target_sources)) != before
    if not targets and not target_sources:
        return {"stamps": 0, "sources": 0}
    writer = visa.agent if visa else HOLDER
    stamp = {"op": "forget", "targets": sorted(targets) or [], "reason": "holder request"}
    if not targets:
        # Sources alone: remove them; nothing in the log referenced them.
        passport.sources = [s for s in passport.sources if s.id not in target_sources]
        passport.touch(now)
        return {"stamps": 0, "sources": len(target_sources)}
    result = _propose_forget(passport, stamp, sorted(target_sources), writer, visa, now)
    if not result.accepted:
        return {"stamps": 0, "sources": 0, "quarantined": len(result.quarantined)}
    return {"stamps": len(targets), "sources": len(target_sources & set(sources))}


def _propose_forget(passport, item, erase_sources, writer, visa, now):
    moment = now or utcnow()
    clock, latest = _clock_for(passport, moment, "local")
    stamp = Stamp(op="forget", targets=item["targets"], author=writer, visa=visa.id if visa else "",
                  origin="user_direct", reason=item["reason"], erase_sources=erase_sources,
                  hlc=clock.tick(), seen=latest)
    result = ProposeResult()
    verdict, reason = admit(passport, stamp, visa, now=moment)
    if verdict == REJECTED:
        result.rejected.append({"index": 0, "reason": reason})
        return result
    stamp.id = stamp_id(stamp)
    stamp.admission, stamp.admission_reason = verdict, reason
    if passport._signer is not None:
        stamp.sig = passport._signer.sign(stamp)
    passport.stamps.append(stamp)
    if verdict == "accepted":
        _erase(passport, stamp)
        result.accepted.append({"id": stamp.id, "op": "forget", "section": "", "summary": "forget"})
    else:
        result.quarantined.append({"id": stamp.id, "op": "forget", "reason": reason})
    project(passport)
    passport.touch(moment)
    return result


# -- merging -------------------------------------------------------------------

def _merge_json(a, b):
    """Commutative merge of JSON-like metadata."""
    if a == b:
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _merge_json(a[k], b[k]) if k in a and k in b else (a[k] if k in a else b[k])
                for k in sorted(set(a) | set(b))}
    return a if canonical(a) >= canonical(b) else b


def _pick(a, b):
    return a if canonical(a.to_compact()) >= canonical(b.to_compact()) else b


def merge(left: Passport, right: Passport) -> Passport:
    """Union two replicas. Commutative, associative and idempotent on the log."""
    a, b = left.model_copy(deep=True), right.model_copy(deep=True)
    for p in (a, b):
        p.assert_supported()
        ensure_log(p)
    if a.user_key and b.user_key and a.user_key != b.user_key:
        raise ValueError("Cannot merge passports of different users")
    stamps = union(a.stamps, b.stamps)
    erased = {sid for s in stamps if s.op == "forget" and s.admission == "accepted" for sid in s.erase_sources}
    sources = {}
    for s in a.sources + b.sources:
        sources[s.id] = _pick(sources[s.id], s) if s.id in sources else s
    agents = {}
    for record in list(a.agents.values()) + list(b.agents.values()):
        if record.id not in agents:
            agents[record.id] = record
            continue
        old = agents[record.id]
        merged = _pick(old, record).model_copy(deep=True)
        firsts = [x for x in (old.first_seen, record.first_seen) if x]
        lasts = [x for x in (old.last_seen, record.last_seen) if x]
        merged.first_seen = min(firsts) if firsts else None
        merged.last_seen = max(lasts) if lasts else None
        agents[record.id] = merged
    visas = {}
    for visa in a.visas + b.visas:
        if visa.id not in visas:
            visas[visa.id] = visa
            continue
        old = visas[visa.id]
        merged = _pick(old, visa).model_copy(deep=True)
        revoked = [x for x in (old.revoked_at, visa.revoked_at) if x]
        merged.revoked_at = min(revoked) if revoked else None
        visas[visa.id] = merged
    out = Passport(version=a.version, updated_at=max(a.updated_at, b.updated_at),
                   user_key=a.user_key or b.user_key, holder=_merge_json(a.holder, b.holder),
                   agents={k: agents[k] for k in sorted(agents)},
                   visas=[visas[k] for k in sorted(visas)],
                   stamps=stamps,
                   sources=[sources[k] for k in sorted(sources) if k not in erased],
                   meta=_merge_json(a.meta, b.meta))
    if stamps:
        project(out)
    else:
        for section in STAMP_SECTIONS:
            setattr(out, section, getattr(a, section))
    return out


# -- reading the log ---------------------------------------------------------------

def history(passport: Passport, *, subject: str = "", key: str | None = None, item: str | None = None,
            section: str | None = None) -> list[dict[str, Any]]:
    """Every stamp ever written to one slot, oldest first, with what became of it."""
    section = section or ("open_loops" if item else "facts")
    from .resolver import _loop_key
    want_subject = subject.casefold() if subject.casefold() not in _HOLDER_WORDS else ""
    if not passport.stamps:
        rows = []
        for r in getattr(passport, section):
            if section in {"facts", "prefs"} and (r.subject.casefold() != want_subject
                                                  or normalize_key(r.key) != normalize_key(key or "")):
                continue
            rows.append({"value": r.value if hasattr(r, "value") else r.to_compact(), "status": r.status,
                         "time": r.ts.isoformat(), "author": r.author or r.source})
        return rows
    state = analyse(passport.stamps)
    current = {sid: r.status for r in getattr(passport, section) for sid in r.stamp_ids}
    rows = []
    for s in passport.stamps:
        if s.section != section:
            continue
        claim = s.claim
        if not s.redacted:
            if str(claim.get("subject", "")).casefold() != want_subject:
                continue
            if section in {"facts", "prefs"} and normalize_key(str(claim.get("key", ""))) != normalize_key(key or ""):
                continue
            if section == "open_loops" and _loop_key(claim.get("item", "")) != _loop_key(item or ""):
                continue
        else:
            continue
        if s.admission == "quarantined" and s.id not in state.confirmed:
            status = "quarantined"
        elif s.id in state.forgotten:
            status = "forgotten"
        elif s.id in state.corrected:
            status = "corrected"
        elif s.id in state.retracted:
            status = "retracted"
        elif s.id in state.superseded:
            status = "superseded"
        elif s.id in current:
            status = "disputed" if current[s.id] == "disputed" else "current"
        else:
            status = "history"
        rows.append({"id": s.id, "op": s.op, "value": claim.get("value", claim.get("status")),
                     "time": claim_time(s).isoformat(), "author": s.author, "origin": state.origin(s),
                     "status": status, "valid_from": claim.get("valid_from", ""),
                     "valid_until": state.expires.get(s.id, claim.get("valid_until", ""))})
    rows.sort(key=lambda r: (r["time"], r["id"]))
    return rows


def verify(passport: Passport, *, signer=None) -> dict[str, Any]:
    """Offline integrity: content addresses, targets, head, signatures, projection, secrets."""
    issues = []
    ids = [s.id for s in passport.stamps]
    if len(ids) != len(set(ids)):
        issues.append({"code": "duplicate_stamp"})
    known = set(ids)
    for s in passport.stamps:
        if not s.redacted and stamp_id(s) != s.id:
            issues.append({"code": "id_mismatch", "stamp": s.id})
        for t in s.targets:
            if t not in known:
                issues.append({"code": "missing_target", "stamp": s.id, "target": t})
        if not s.redacted and scan_secret(s.claim, s.evidence):
            issues.append({"code": "secret", "stamp": s.id})
        if signer is not None and not s.author.startswith("import:") and not signer.verify(s):
            issues.append({"code": "bad_signature" if s.sig else "unsigned", "stamp": s.id})
    head = merkle_root(ids)
    if passport._loaded_head and passport._loaded_head != head:
        issues.append({"code": "head_mismatch", "stored": passport._loaded_head, "computed": head})
    if passport.stamps:
        fresh = passport.model_copy(deep=True)
        project(fresh)
        for section in STAMP_SECTIONS:
            mine = [r.to_compact() for r in getattr(passport, section)]
            theirs = [r.to_compact() for r in getattr(fresh, section)]
            if canonical(mine) != canonical(theirs):
                issues.append({"code": "projection_drift", "section": section})
    return {"valid": not issues, "issues": issues, "stamps": len(ids), "head": head}


def apply_update(passport: Passport, update: dict[str, list], *, author: str, origin: str = "agent_inferred",
                 now=None) -> ProposeResult:
    """Write an extraction update as stamps (the 2.x ingest path on a log-backed passport)."""
    items = []
    for section in ("facts", "prefs", "entities", "open_loops", "episodes", "observations"):
        for record in update.get(section) or []:
            claim = record.to_compact()
            claim.pop("stamp_ids", None)
            items.append({"op": "assert", "section": section, "claim": claim, "origin": origin})
    return propose(passport, items, author=author, origin=origin, now=now) if items else ProposeResult()


def set_loop_status(passport: Passport, item: str, status: str, *, subject: str = "", now=None) -> ProposeResult:
    from .resolver import _loop_key
    rows = [r for r in passport.open_loops if _loop_key(r.item) == _loop_key(item)
            and r.subject.casefold() == subject.casefold()]
    if not rows:
        raise KeyError(item)
    current = rows[0]
    claim = {**current.to_compact(), "status": status, "ts": (now or utcnow()).isoformat()}
    for key in ("stamp_ids", "author", "origin"):
        claim.pop(key, None)
    return propose(passport, [{"op": "supersede", "section": "open_loops", "claim": claim,
                               "targets": current.stamp_ids}], author=HOLDER, now=now)


def rank(origin: str) -> int:
    return ORIGIN_RANK.get(origin, 0)
