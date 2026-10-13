"""
Passport stores: a locked file on disk, or any backend with the same two calls.

    store = FileStore("holder.json")
    with store.transaction() as passport:      # load, lock, mutate, save
        passport.propose([...], visa=visa)
    passport = store.read()

Writers on one machine serialise through an exclusive lock file. Replicas on
different machines merge their stamp logs instead (`Passport.merge`). The
Firestore store uses optimistic concurrency, and a write that loses the race
merges with the winner instead of failing: the stamp log is a CRDT.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .schema import Passport

_LOCKS: dict[str, threading.Lock] = {}


class StoreBusy(RuntimeError):
    pass


class FileStore:
    def __init__(self, path, *, holder: str = "holder", timeout: float = 10.0):
        self.path = Path(path)
        self.holder = holder
        self.timeout = timeout
        self._local = _LOCKS.setdefault(str(self.path.resolve()), threading.Lock())

    def exists(self) -> bool:
        return self.path.exists()

    def version(self) -> str:
        return str(self.path.stat().st_mtime_ns) if self.path.exists() else ""

    def read(self) -> Passport:
        if not self.path.exists():
            return Passport.issue(self.holder)
        return Passport.load(self.path)

    @contextmanager
    def _lock(self):
        lock_path = self.path.with_name(self.path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        with self._local:
            while True:
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    break
                except FileExistsError:
                    # A lock older than a minute belongs to a crashed writer.
                    try:
                        if time.time() - lock_path.stat().st_mtime > 60:
                            lock_path.unlink()
                            continue
                    except FileNotFoundError:
                        continue
                    if time.monotonic() > deadline:
                        raise StoreBusy(f"{self.path} is locked by another writer") from None
                    time.sleep(0.05)
            try:
                os.write(fd, str(os.getpid()).encode("ascii"))
                yield
            finally:
                os.close(fd)
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    @contextmanager
    def transaction(self, signer=None):
        with self._lock():
            passport = self.read()
            passport._signer = signer
            yield passport
            passport._signer = None
            passport.save(self.path)


#: Firestore caps a document at 1 MiB; compressed JSON keeps a large passport inside it.
_FIRESTORE_LIMIT = 1_000_000


def encode(passport: Passport) -> dict[str, Any]:
    raw = zlib.compress(passport.to_json(indent=None).encode("utf-8"), 9)
    blob = base64.b64encode(raw).decode("ascii")
    if len(blob) > _FIRESTORE_LIMIT:
        raise ValueError("passport exceeds one Firestore document; retain fewer source turns")
    return {"format": "sozograph/3.1+zlib", "z": blob, "updated_at": passport.updated_at.isoformat()}


def decode(data: dict[str, Any] | None) -> Passport | None:
    if not data:
        return None
    if "z" in data:
        return Passport.from_json(zlib.decompress(base64.b64decode(data["z"])).decode("utf-8"))
    if "json" in data:
        return Passport.from_json(data["json"])
    return Passport.from_dict(data)


class FirestoreStore:
    """
    One passport in one Firestore document.

    `transaction()` reads the document with its update time, lets the caller
    mutate the passport, then writes on the condition that nobody else wrote
    in between. When someone did, it merges the two passports and tries again.
    """

    def __init__(self, project: str | None, collection: str, document: str, *, client=None,
                 holder: str = "holder", attempts: int = 5, database: str | None = None):
        if client is None:
            try:
                from google.cloud import firestore
            except ImportError as exc:  # pragma: no cover - exercised without the extra
                raise ImportError('The Firestore store needs: pip install "sozograph[firestore]"') from exc
            client = firestore.Client(project=project, database=database) if database else \
                firestore.Client(project=project)
        self.client = client
        self.ref = client.collection(collection).document(document)
        self.holder = holder
        self.attempts = attempts
        self.path = None

    def version(self) -> str:
        snap = self.ref.get()
        return str(snap.update_time) if snap.exists else ""

    def exists(self) -> bool:
        return self.ref.get().exists

    def read(self) -> Passport:
        snap = self.ref.get()
        return (decode(snap.to_dict()) if snap.exists else None) or Passport.issue(self.holder)

    @contextmanager
    def transaction(self, signer=None):
        snap = self.ref.get()
        passport = (decode(snap.to_dict()) if snap.exists else None) or Passport.issue(self.holder)
        passport._signer = signer
        yield passport
        passport._signer = None
        for _ in range(self.attempts):
            payload = encode(passport)
            try:
                if snap.exists:
                    self.ref.update(payload, option=self.client.write_option(last_update_time=snap.update_time))
                else:
                    self.ref.create(payload)
                return
            except Exception as exc:  # FailedPrecondition, AlreadyExists, Conflict
                if type(exc).__name__ not in {"FailedPrecondition", "AlreadyExists", "Conflict", "Aborted"}:
                    raise
                snap = self.ref.get()
                remote = decode(snap.to_dict()) if snap.exists else None
                if remote is not None:
                    passport = Passport.merge(remote, passport)
        raise StoreBusy("the passport kept changing under this write")


class JsonDocument:
    """A small JSON state document (the visa office) in a file or a Firestore document."""

    def __init__(self, path: Path | None = None, ref=None):
        self.path, self.ref = path, ref

    def load(self) -> dict[str, Any]:
        if self.ref is not None:
            snap = self.ref.get()
            return json.loads(snap.to_dict().get("json", "{}")) if snap.exists else {}
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path and self.path.exists() else {}

    def save(self, data: dict[str, Any]) -> None:
        if self.ref is not None:
            self.ref.set({"json": json.dumps(data)})
            return
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def open_store(spec: str, *, client=None):
    """`path/to/file.json` or `firestore://project/collection/document[?database=name]`."""
    if spec.startswith("firestore://"):
        body, _, query = spec[len("firestore://"):].partition("?")
        parts = body.strip("/").split("/")
        if len(parts) != 3:
            raise ValueError("firestore://project/collection/document[?database=name]")
        options = dict(item.split("=", 1) for item in query.split("&") if "=" in item)
        project, collection, document = parts
        store = FirestoreStore(project or None, collection, document, client=client,
                               database=options.get("database") or os.getenv("SOZOGRAPH_FIRESTORE_DATABASE"))
        office = JsonDocument(ref=store.client.collection(collection).document(document + "__office"))
        return store, office
    path = Path(spec)
    return FileStore(path), JsonDocument(path=path.with_name(path.name + ".office.json"))
