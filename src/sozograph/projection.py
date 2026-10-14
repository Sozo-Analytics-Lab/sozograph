"""
Belief projection: the sections of a log-backed passport, computed from its stamps.

The projection is a pure function of the stamp log and the agent registry, so
every replica holding the same stamps holds the same beliefs. It inherits the
tuned 2.x resolver instead of rebuilding retrieval-critical behavior: live
stamps fold through the same upsert, deduplication and contradiction logic in
claim-time order, which makes a single-writer log reproduce 2.x output exactly.

Three rules from Passport 3.1 sit on top of the fold:

* Explicit operations. A supersede closes its target and records the change;
  a correct removes its target from history; a retract ends it; a forget
  erases it; an expire sets its end date; a holder confirmation admits a
  quarantined stamp and raises its origin.
* The origin guard. A third-party or tool-result claim cannot overtake a
  higher-trust claim by being newer. Both stay on the frontier.
* Competence. Frontier alternatives are scored in log-odds units by the
  author's competence in the claim's domain plus an origin weight. Scores
  within DELTA of the best stay disputed; the rest stay in history.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .resolver import (
    ResolveStats,
    _add_observation,
    _change,
    _loop_key,
    _merge_entities,
    _record_contradiction,
    _sort,
    _upsert_kv,
    _value_equal,
)
from .schema import (
    ORIGIN_RANK,
    Contradiction,
    Entity,
    Episode,
    Fact,
    Observation,
    OpenLoop,
    Passport,
    Preference,
    Stamp,
)
from .shelf import parse_moment
from .stamps import hlc_datetime
from .trust import Competence
from .utils import normalize_key, sha256_json

MODELS = {"facts": Fact, "prefs": Preference, "entities": Entity, "open_loops": OpenLoop,
          "episodes": Episode, "observations": Observation, "contradictions": Contradiction}
ORIGIN_WEIGHT = {"user_direct": 3.0, "user_confirmed": 3.0, "agent_inferred": 0.0,
                 "imported": -0.25, "tool_result": -0.5, "third_party": -1.0}
DELTA = 0.5
CLAIMING = {"assert", "supersede", "correct"}
_GOVERNANCE = ("stamp_ids", "author", "origin", "domains", "kind", "valid_from", "valid_until", "sensitivity")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass
class LogState:
    by_id: dict[str, Stamp]
    confirmed: set[str] = field(default_factory=set)
    corrected: set[str] = field(default_factory=set)
    retracted: set[str] = field(default_factory=set)
    forgotten: set[str] = field(default_factory=set)
    superseded: dict[str, list[Stamp]] = field(default_factory=lambda: defaultdict(list))
    expires: dict[str, str] = field(default_factory=dict)
    live: list[Stamp] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)

    def admitted(self, stamp: Stamp) -> bool:
        return stamp.admission == "accepted" or stamp.id in self.confirmed

    def origin(self, stamp: Stamp) -> str:
        if stamp.id in self.confirmed and ORIGIN_RANK.get(stamp.origin, 0) < ORIGIN_RANK["user_confirmed"]:
            return "user_confirmed"
        return stamp.origin


def analyse(stamps: list[Stamp]) -> LogState:
    state = LogState(by_id={s.id: s for s in stamps})
    for s in stamps:
        if s.op == "confirm" and s.author == "holder" and s.admission == "accepted":
            state.confirmed.update(s.targets)
    controls = sorted((s for s in stamps if state.admitted(s)), key=lambda s: (s.hlc, s.id))
    for s in controls:
        if s.op == "correct":
            state.corrected.update(s.targets)
        elif s.op == "retract":
            state.retracted.update(s.targets)
        elif s.op == "forget":
            state.forgotten.update(s.targets)
        elif s.op == "supersede":
            for t in s.targets:
                state.superseded[t].append(s)
        elif s.op == "expire" and s.claim.get("valid_until"):
            for t in s.targets:
                state.expires[t] = str(s.claim["valid_until"])
    state.forgotten.update(s.id for s in stamps if s.redacted)
    gone = state.corrected | state.retracted | state.forgotten
    state.live = [s for s in stamps
                  if s.op in CLAIMING and s.claim and not s.redacted and state.admitted(s)
                  and s.id not in gone and s.section in MODELS]
    return state


def claim_time(stamp: Stamp) -> datetime:
    """When the claim became true: valid_from, else discussion time, else the clock."""
    claim = stamp.claim
    for key in ("valid_from", "ts"):
        moment = parse_moment(claim.get(key))
        if moment is not None:
            return moment
    return hlc_datetime(stamp.hlc) if stamp.hlc else _EPOCH


def _order(stamp: Stamp):
    return (claim_time(stamp), stamp.hlc, stamp.id)


def record_from(stamp: Stamp, state: LogState):
    """Validate a stamp's claim into its section model, with governance fields."""
    model = MODELS[stamp.section]
    data = dict(stamp.claim)
    data.pop("section", None)
    if stamp.section in {"facts", "prefs"} and data.get("valid_from"):
        # A 3.1 belief's time is when it became true; discussion time stays on the stamp.
        data["ts"] = data["valid_from"]
    data["stamp_ids"] = [stamp.id]
    data.setdefault("author", stamp.author)
    data["origin"] = state.origin(stamp)
    if stamp.id in state.expires:
        data["valid_until"] = state.expires[stamp.id]
    try:
        return model.model_validate(data)
    except Exception:
        state.invalid.append(stamp.id)
        return None


def _slot(stamp: Stamp) -> tuple:
    claim = stamp.claim
    if stamp.section == "open_loops":
        return (str(claim.get("subject", "")).casefold(), _loop_key(claim.get("item", "")))
    return (str(claim.get("subject", "")).casefold(), normalize_key(str(claim.get("key", ""))))


def _rank(stamp: Stamp, state: LogState) -> int:
    return ORIGIN_RANK.get(state.origin(stamp), 0)


def _score(stamp: Stamp, state: LogState, competence: Competence) -> float:
    domains = stamp.claim.get("domains") or ["general"]
    return competence.logit(stamp.author, domains[0]) + ORIGIN_WEIGHT.get(state.origin(stamp), 0.0)


def _classes(stamps: list[Stamp]) -> list[list[Stamp]]:
    classes: list[list[Stamp]] = []
    for s in sorted(stamps, key=_order):
        for c in classes:
            if _value_equal(c[0].claim.get("value"), s.claim.get("value")):
                c.append(s)
                break
        else:
            classes.append([s])
    return classes


def _winners(classes, state, competence):
    """Frontier under the origin guard, then competence scoring."""
    def ctime(c):
        return max(claim_time(s) for s in c)

    def rank(c):
        return max(_rank(s, state) for s in c)

    def dominates(b, a):
        return ctime(b) > ctime(a) and (rank(b) >= rank(a) or rank(b) >= ORIGIN_RANK["imported"])

    frontier = [c for c in classes if not any(dominates(b, c) for b in classes if b is not c)]
    if not frontier:
        frontier = [max(classes, key=lambda c: (ctime(c), rank(c), max(s.hlc for s in c)))]
    if len(frontier) == 1:
        return frontier, []
    scored = sorted(((max(_score(s, state, competence) for s in c), c) for c in frontier),
                    key=lambda x: (-x[0], sha256_json(x[1][0].claim.get("value"))))
    best = scored[0][0]
    winners = [c for score, c in scored if best - score < DELTA]
    losers = [c for score, c in scored if best - score >= DELTA]
    return winners, losers


def _govern(record, members: list[Stamp], state: LogState) -> None:
    latest = max(members, key=_order)
    record.stamp_ids = sorted({s.id for s in members} | set(record.stamp_ids))
    record.author = latest.author
    record.origin = max((state.origin(s) for s in members), key=lambda o: (ORIGIN_RANK.get(o, 0), o))
    record.domains = sorted({d for s in members for d in (s.claim.get("domains") or [])})
    record.kind = latest.claim.get("kind", "") or ""
    record.valid_from = str(latest.claim.get("valid_from", "") or "")
    record.valid_until = state.expires.get(latest.id, str(latest.claim.get("valid_until", "") or ""))
    levels = [s.claim.get("sensitivity") for s in members if s.claim.get("sensitivity") is not None]
    record.sensitivity = max(levels) if levels else None


def _project_kv(section, stamps, state, competence, contradictions, diagnostics):
    items_out = []
    slots: dict[tuple, list[Stamp]] = defaultdict(list)
    for s in stamps:
        slots[_slot(s)].append(s)
    for key in sorted(slots):
        members = slots[key]
        current = [s for s in members if s.id not in state.superseded]
        if not current:
            continue
        classes = _classes(current)
        winners, losers = _winners(classes, state, competence)
        horizon = max(claim_time(s) for c in winners for s in c)
        loser_ids = {s.id for c in losers for s in c}
        fold = [s for s in current if s.id not in loser_ids and claim_time(s) <= horizon]
        diagnostics["alternatives"].extend(sorted(loser_ids | {s.id for s in current
                                                               if s.id not in loser_ids and claim_time(s) > horizon}))
        items: list = []
        local: list[Contradiction] = []
        stats = ResolveStats()
        for s in sorted(fold, key=_order):
            record = record_from(s, state)
            if record is None:
                continue
            _upsert_kv(items=items, incoming=record, contradictions=local, stats=stats)
        winner_values = [c[0].claim.get("value") for c in winners]
        if len(winners) == 1:
            keep = [r for r in items if _value_equal(r.value, winner_values[0])]
            dropped = [r for r in items if r not in keep]
            if dropped:
                local = [c for c in local if not (c.status == "disputed" and any(
                    _value_equal(c.old, d.value) or _value_equal(c.new, d.value) for d in dropped))]
                for r in keep:
                    r.status = "active"
            items = keep
        else:
            present = [r for r in items if any(_value_equal(r.value, v) for v in winner_values)]
            for c in winners:
                if not any(_value_equal(r.value, c[0].claim.get("value")) for r in present):
                    record = record_from(max(c, key=_order), state)
                    if record is not None:
                        present.append(record)
            for r in present:
                r.status = "disputed"
            ordered = sorted(present, key=lambda r: sha256_json(r.value))
            for left, right in zip(ordered, ordered[1:], strict=False):
                _record_contradiction(local, _change(left, right, disputed=True))
            items = present
        by_value = {}
        for c in classes:
            by_value[sha256_json(c[0].claim.get("value"))] = [s for s in c if s.id in {f.id for f in fold}] or c
        for record in items:
            group = next((g for k, g in by_value.items() if _value_equal(g[0].claim.get("value"), record.value)), None)
            if group:
                _govern(record, group, state)
        for change in local:
            _govern_change(change, classes, state)
            _record_contradiction(contradictions, change)
        for target_id in sorted(s.id for s in members if s.id in state.superseded):
            target = state.by_id[target_id]
            old = record_from(target, state)
            for successor in state.superseded[target_id]:
                if successor.id in state.forgotten or not successor.claim or successor.section != section:
                    continue
                new = record_from(successor, state)
                if old is None or new is None or _value_equal(old.value, new.value):
                    continue
                change = _change(old, new)
                change.stamp_ids = sorted({target.id, successor.id})
                change.domains = sorted(set(old.domains or target.claim.get("domains") or [])
                                        | set(new.domains or successor.claim.get("domains") or []))
                _record_contradiction(contradictions, change)
        items_out.extend(items)
    return items_out


def _govern_change(change: Contradiction, classes, state) -> None:
    ids, domains, levels = set(), set(), []
    for c in classes:
        value = c[0].claim.get("value")
        if _value_equal(value, change.old) or _value_equal(value, change.new):
            for s in c:
                ids.add(s.id)
                domains.update(s.claim.get("domains") or [])
                if s.claim.get("sensitivity") is not None:
                    levels.append(s.claim["sensitivity"])
    change.stamp_ids = sorted(ids)
    change.domains = sorted(domains)
    change.sensitivity = max(levels) if levels else None


def _project_loops(stamps, state):
    slots: dict[tuple, list[Stamp]] = defaultdict(list)
    for s in stamps:
        if s.id not in state.superseded:
            slots[_slot(s)].append(s)
    out = []
    for key in sorted(slots):
        group = sorted(slots[key], key=_order)
        chosen, members = None, []
        for s in group:
            record = record_from(s, state)
            if record is None:
                continue
            members.append(s)
            if chosen is None or record.ts > chosen.ts:
                chosen = record
        if chosen is not None:
            winner = next(s for s in reversed(members) if s.id in chosen.stamp_ids)
            _govern(chosen, [winner], state)
            chosen.stamp_ids = sorted(s.id for s in members)
            out.append(chosen)
    return out


def project(passport: Passport) -> Passport:
    """Recompute every claim section of a log-backed passport from its stamps."""
    if not passport.stamps:
        return passport
    state = analyse(passport.stamps)
    competence = Competence(passport.stamps, passport.agents)
    by_section: dict[str, list[Stamp]] = defaultdict(list)
    for s in state.live:
        by_section[s.section].append(s)
    diagnostics = {"alternatives": [], "invalid": []}
    contradictions: list[Contradiction] = []
    for s in sorted(by_section.get("contradictions", []), key=_order):
        record = record_from(s, state)
        if record is not None:
            _record_contradiction(contradictions, record)
    passport.facts = _project_kv("facts", by_section.get("facts", []), state, competence, contradictions, diagnostics)
    passport.prefs = _project_kv("prefs", by_section.get("prefs", []), state, competence, contradictions, diagnostics)
    passport.open_loops = _project_loops(by_section.get("open_loops", []), state)
    entities_holder = Passport()
    entity_records = [r for r in (record_from(s, state) for s in sorted(by_section.get("entities", []),
                                                                          key=lambda s: (s.hlc, s.id))) if r]
    _merge_entities(entities_holder, entity_records, ResolveStats())
    passport.entities = entities_holder.entities
    episodes: dict[str, Episode] = {}
    for s in sorted(by_section.get("episodes", []), key=lambda s: (s.hlc, s.id)):
        record = record_from(s, state)
        if record is None:
            continue
        earlier = episodes.get(record.id)
        if earlier is not None:
            record.stamp_ids = sorted(set(earlier.stamp_ids) | set(record.stamp_ids))
        episodes[record.id] = record
    passport.episodes = list(episodes.values())
    observations: list[Observation] = []
    for s in sorted(by_section.get("observations", []), key=_order):
        record = record_from(s, state)
        if record is not None:
            _add_observation(observations, record)
    passport.observations = observations
    passport.contradictions = contradictions
    _sort(passport)
    diagnostics["invalid"] = sorted(set(state.invalid))
    passport.meta.pop("projection", None)
    if diagnostics["invalid"]:
        passport.meta["projection"] = {"invalid": diagnostics["invalid"]}
    return passport


def strip_governance(data: dict) -> dict:
    """A record's 2.x view: governance fields removed (for parity checks)."""
    return {k: v for k, v in data.items() if k not in _GOVERNANCE}
