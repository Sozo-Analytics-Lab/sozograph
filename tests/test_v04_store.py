"""Stores: the locked file, and Firestore with merge-on-conflict, against a fake client."""
from __future__ import annotations

import itertools
import threading

from sozograph import Passport
from sozograph.store import FileStore, FirestoreStore, decode, encode, open_store


class FailedPrecondition(Exception):
    pass


class AlreadyExists(Exception):
    pass


class Snap:
    def __init__(self, data, update_time):
        self._data, self.update_time = data, update_time
        self.exists = data is not None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class Ref:
    clock = itertools.count(1)

    def __init__(self):
        self.data, self.time = None, None

    def get(self):
        return Snap(self.data, self.time)

    def create(self, payload):
        if self.data is not None:
            raise AlreadyExists()
        self.data, self.time = dict(payload), next(Ref.clock)

    def update(self, payload, option=None):
        if option is not None and option["last_update_time"] != self.time:
            raise FailedPrecondition()
        self.data, self.time = dict(payload), next(Ref.clock)

    def set(self, payload):
        self.data, self.time = dict(payload), next(Ref.clock)


class Client:
    def __init__(self):
        self.docs = {}

    def collection(self, name):
        client = self

        class Collection:
            def document(self, doc):
                return client.docs.setdefault((name, doc), Ref())

        return Collection()

    def write_option(self, last_update_time):
        return {"last_update_time": last_update_time}


def test_encoding_is_compressed_and_lossless():
    p = Passport.issue("rairo")
    p.propose([{"attribute": "note", "value": "x" * 5000}])
    data = encode(p)
    assert len(data["z"]) < 2500 and decode(data).content_hash() == p.content_hash()


def test_concurrent_firestore_writes_merge_instead_of_failing():
    client = Client()
    store = FirestoreStore("proj", "passports", "rairo", client=client)
    with store.transaction() as p:
        p.propose([{"attribute": "home_city", "value": "Harare"}])
    first = store.transaction()
    a = first.__enter__()
    with store.transaction() as b:
        b.propose([{"attribute": "editor", "value": "VS Code"}])
    a.propose([{"attribute": "shoe_size", "value": "44"}])
    first.__exit__(None, None, None)
    final = store.read()
    assert {f.key for f in final.facts} == {"home_city", "editor", "shoe_size"}
    assert final.verify()["valid"]


def test_open_store_parses_firestore_specs():
    client = Client()
    store, office = open_store("firestore://proj/passports/rairo", client=client)
    assert isinstance(store, FirestoreStore)
    office.save({"clients": {}})
    assert office.load() == {"clients": {}}


def test_file_store_serialises_threads(tmp_path):
    store = FileStore(tmp_path / "p.json")
    with store.transaction() as p:
        p.ensure_log()

    def write(i):
        with store.transaction() as p:
            p.propose([{"attribute": f"key_{i}", "value": i}])

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.read().facts) == 8
    assert not (tmp_path / "p.json.lock").exists()


def test_ed25519_author_signatures_verify_offline():
    import pytest
    pytest.importorskip("cryptography")
    from sozograph.signing import Ed25519Signer
    signer = Ed25519Signer(key_id="claude-code")
    p = Passport.issue()
    p._signer = signer
    p.propose([{"attribute": "editor", "value": "VS Code"}])
    assert p.verify(signer=signer.verifier())["valid"]
    other = Ed25519Signer(key_id="claude-code")
    assert not p.verify(signer=other.verifier())["valid"]


def test_firestore_spec_takes_a_database():
    from sozograph.store import open_store
    client = Client()
    store, _ = open_store("firestore://proj/passports/rairo?database=sozograph", client=client)
    assert isinstance(store, FirestoreStore)
