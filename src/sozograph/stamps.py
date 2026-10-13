"""
Stamp primitives: content addresses, a hybrid logical clock, the Merkle head,
attestation, and the join that makes the log a CRDT.

A stamp log is a map from stamp id to payload. Merge is the union of ids and
the join of payloads, where a redacted payload sits above every other and a
quarantined verdict above an accepted one. Union and a lattice join are each
commutative, associative and idempotent, so any two replicas that hold the same
stamps hold the same log, in any delivery order. Everything here is stdlib.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from collections.abc import Iterable
from datetime import datetime, timezone

from .schema import Stamp

#: Fields outside the content address: the replica's verdict, its attestation,
#: and the holder's erasure.
_UNHASHED = {"id", "admission", "admission_reason", "sig", "redacted"}


def canonical(data) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str)


def stamp_id(stamp: Stamp) -> str:
    """`st_` plus the first 128 bits of SHA-256 over the canonical content."""
    body = {k: v for k, v in stamp.model_dump(mode="json").items() if k not in _UNHASHED}
    return "st_" + hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()[:32]


def merkle_root(ids: Iterable[str]) -> str:
    """SHA-256 Merkle root over the sorted stamp ids; empty log hashes the empty string."""
    level = [hashlib.sha256(i.encode("utf-8")).digest() for i in sorted(set(ids))]
    if not level:
        return hashlib.sha256(b"").hexdigest()
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


# -- hybrid logical clock -----------------------------------------------------

def hlc_encode(wall_ms: int, counter: int, node: str) -> str:
    """Fixed-width so lexicographic order is clock order."""
    return f"{wall_ms:013d}-{counter:05d}-{node}"


def hlc_wall(hlc: str) -> int:
    try:
        return int(hlc.split("-", 1)[0])
    except (ValueError, IndexError):
        return 0


def hlc_datetime(hlc: str) -> datetime:
    return datetime.fromtimestamp(hlc_wall(hlc) / 1000, tz=timezone.utc)


class Clock:
    """
    Hybrid logical clock (Kulkarni et al., 2014).

    Readings never go backwards on one node, even when the wall clock does, and
    `observe` folds in readings from other replicas so causality survives a merge.
    """

    def __init__(self, node: str = "local", now=None):
        self.node = node.replace("-", "_") or "local"
        self._now = now
        self._last = (0, 0)
        self._lock = threading.Lock()

    def _wall(self) -> int:
        if self._now is not None:
            moment = self._now()
            return int(moment.timestamp() * 1000)
        return int(time.time() * 1000)

    def observe(self, hlc: str) -> None:
        if not hlc:
            return
        wall = hlc_wall(hlc)
        try:
            counter = int(hlc.split("-")[1])
        except (IndexError, ValueError):
            counter = 0
        with self._lock:
            if (wall, counter) > self._last:
                self._last = (wall, counter)

    def tick(self) -> str:
        with self._lock:
            wall = self._wall()
            last_wall, last_counter = self._last
            if wall > last_wall:
                self._last = (wall, 0)
            else:
                self._last = (last_wall, last_counter + 1)
            return hlc_encode(*self._last, self.node)


# -- attestation ----------------------------------------------------------------

class HmacSigner:
    """
    Server attestation: binds a stamp id to the replica that admitted it.

    SMSR (2026) showed that a signature boundary on memory writes stops every
    unsigned injection. A hosted passport signs each admitted stamp; `verify`
    then rejects stamps that entered the log some other way.
    """

    scheme = "hmac-sha256"

    def __init__(self, key: bytes | str, key_id: str = "server"):
        self.key = key.encode("utf-8") if isinstance(key, str) else key
        self.key_id = key_id

    def sign(self, stamp: Stamp) -> str:
        mac = hmac.new(self.key, stamp.id.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{self.scheme}:{self.key_id}:{mac}"

    def verify(self, stamp: Stamp) -> bool:
        if not stamp.sig.startswith(f"{self.scheme}:{self.key_id}:"):
            return False
        return hmac.compare_digest(stamp.sig, self.sign(stamp))


# -- the join ---------------------------------------------------------------

def redact(stamp: Stamp) -> Stamp:
    """Erase a payload, keeping the identity, op, targets and clock it needs to stay ordered."""
    out = stamp.model_copy(deep=True)
    out.claim = {}
    out.evidence = []
    out.reason = ""
    out.redacted = True
    return out


def join(a: Stamp, b: Stamp) -> Stamp:
    """Join two copies of one stamp. Redaction, then quarantine, then the larger signature win."""
    if a.id != b.id:
        raise ValueError("join requires the same stamp id")
    if a.redacted or b.redacted:
        base = a if a.redacted else b
        other = b if base is a else a
        out = base.model_copy(deep=True)
        out.admission = "quarantined" if "quarantined" in (a.admission, b.admission) else out.admission
        out.sig = max(a.sig, b.sig)
        if other.redacted:
            out.erase_sources = sorted(set(a.erase_sources) | set(b.erase_sources))
        return out
    out = a.model_copy(deep=True)
    if b.admission == "quarantined" and a.admission != "quarantined":
        out.admission, out.admission_reason = b.admission, b.admission_reason
    elif a.admission == b.admission:
        out.admission_reason = max(a.admission_reason, b.admission_reason)
    out.sig = max(a.sig, b.sig)
    return out


def union(*logs: Iterable[Stamp]) -> list[Stamp]:
    """Merge stamp logs, then redact every target of an accepted forget stamp."""
    merged: dict[str, Stamp] = {}
    for log in logs:
        for stamp in log:
            merged[stamp.id] = join(merged[stamp.id], stamp) if stamp.id in merged else stamp.model_copy(deep=True)
    forgotten = {t for s in merged.values() if s.op == "forget" and s.admission == "accepted" for t in s.targets}
    for sid in forgotten:
        if sid in merged and not merged[sid].redacted:
            merged[sid] = redact(merged[sid])
    return sorted(merged.values(), key=lambda s: (s.hlc, s.id))
