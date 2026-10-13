"""
Author signatures with Ed25519 (optional extra: `pip install "sozograph[sign]"`).

HMAC attestation proves a stamp entered through one server. An Ed25519
signature proves which writer made it, to anyone holding the public key, with
no shared secret. A replica can then accept stamps from another replica's
writers and check each one offline.
"""
from __future__ import annotations

import base64

from .schema import Stamp


def _crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError('Ed25519 signing needs: pip install "sozograph[sign]"') from exc
    return serialization, Ed25519PrivateKey, Ed25519PublicKey


class Ed25519Signer:
    scheme = "ed25519"

    def __init__(self, private_key=None, *, key_id: str = "author"):
        serialization, private_cls, _ = _crypto()
        self._serialization = serialization
        self.private_key = private_key or private_cls.generate()
        self.key_id = key_id

    @classmethod
    def from_seed(cls, seed: bytes, *, key_id: str = "author") -> Ed25519Signer:
        _, private_cls, _ = _crypto()
        return cls(private_cls.from_private_bytes(seed), key_id=key_id)

    def __deepcopy__(self, memo):
        # Keys are shared by reference; a copied passport signs with the same writer.
        return self

    def public_bytes(self) -> bytes:
        s = self._serialization
        return self.private_key.public_key().public_bytes(s.Encoding.Raw, s.PublicFormat.Raw)

    def sign(self, stamp: Stamp) -> str:
        signature = self.private_key.sign(stamp.id.encode("utf-8"))
        return f"{self.scheme}:{self.key_id}:{base64.urlsafe_b64encode(signature).decode('ascii')}"

    def verifier(self) -> Ed25519Verifier:
        return Ed25519Verifier(self.public_bytes(), key_id=self.key_id)


class Ed25519Verifier:
    scheme = "ed25519"

    def __init__(self, public_key: bytes, *, key_id: str = "author"):
        _, _, public_cls = _crypto()
        self.public_key = public_cls.from_public_bytes(public_key)
        self.key_id = key_id

    def __deepcopy__(self, memo):
        return self

    def verify(self, stamp: Stamp) -> bool:
        prefix = f"{self.scheme}:{self.key_id}:"
        if not stamp.sig.startswith(prefix):
            return False
        try:
            self.public_key.verify(base64.urlsafe_b64decode(stamp.sig[len(prefix):]), stamp.id.encode("utf-8"))
            return True
        except Exception:
            return False
