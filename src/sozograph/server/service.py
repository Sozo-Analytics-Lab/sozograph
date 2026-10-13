"""
The passport service: every operation an agent can request, independent of transport.

MCP tools (stdio and Streamable HTTP) and the REST mirror call the same
methods with the caller's visa. Each write runs in one store transaction, is
signed by the server, and is appended to an audit log.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..protocol import forget as protocol_forget
from ..schema import Passport, Stamp, Visa, utcnow
from ..stamps import HmacSigner, union
from ..store import FileStore
from ..visa import effective_domains, effective_sensitivity, find_visa

_MAX_STAMPS = 50
_SECTIONS = ("facts", "prefs", "open_loops", "observations", "episodes", "entities", "contradictions")


class AccessDenied(PermissionError):
    pass


class PassportService:
    def __init__(self, store: FileStore, *, signer: HmacSigner | None = None, node: str = "server",
                 audit_path: str | Path | None = None, base_url: str = "", audit_stream=None):
        self.store = store
        self.signer = signer
        self.node = node
        self.base_url = base_url.rstrip("/")
        self.audit_path = Path(audit_path) if audit_path else None
        self.audit_stream = audit_stream
        self._cache: tuple[str, Passport] | None = None

    # -- plumbing ------------------------------------------------------------

    def read(self) -> Passport:
        version = self.store.version()
        if self._cache is None or self._cache[0] != version or not version:
            self._cache = (version, self.store.read())
        return self._cache[1]

    def visa_for_token(self, token: str) -> Visa | None:
        return find_visa(self.read(), token) if token else None

    def _audit(self, visa: Visa | None, action: str, detail: dict[str, Any]) -> None:
        row = {"at": utcnow().isoformat(), "visa": visa.id if visa else "holder",
               "agent": visa.agent if visa else "holder", "action": action, **detail}
        if self.audit_stream is not None:
            self.audit_stream.write(json.dumps({"sozograph_audit": row}, ensure_ascii=False) + "\n")
            self.audit_stream.flush()
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _live(passport: Passport, visa: Visa) -> Visa:
        current = next((v for v in passport.visas if v.id == visa.id), None)
        if current is None or not current.active():
            raise AccessDenied("visa is inactive")
        return current

    @staticmethod
    def sees(visa: Visa, claim: dict[str, Any]) -> bool:
        return (visa.sees_domain(effective_domains(claim))
                and effective_sensitivity(claim) <= visa.sensitivity_ceiling)

    # -- reading -------------------------------------------------------------

    def recall(self, visa: Visa, *, query: str = "", budget_tokens: int = 1500) -> dict[str, Any]:
        p = self.read()
        visa = self._live(p, visa)
        result = p.view(visa, query=query or None, budget_chars=max(400, int(budget_tokens) * 4))
        records = [{"section": r.section, "text": r.text, "stamp_ids": r.record.get("stamp_ids", []),
                    "freshness": r.record.get("freshness", "current"), "score": round(r.score, 4)}
                   for r in result.selected]
        self._audit(visa, "recall", {"query": query[:200], "records": len(records)})
        return {"context": result.context, "records": records, "agent": visa.agent,
                "domains": visa.domains, "write": visa.write}

    def history(self, visa: Visa, *, attribute: str = "", subject: str = "", item: str = "") -> list[dict]:
        p = self.read()
        visa = self._live(p, visa)
        rows = p.history(subject=subject, key=attribute or None, item=item or None)
        by_id = {s.id: s for s in p.stamps}
        visible = [r for r in rows if r.get("id") not in by_id or self.sees(visa, by_id[r["id"]].claim)]
        self._audit(visa, "history", {"attribute": attribute, "rows": len(visible)})
        return visible

    def handoff(self, visa: Visa, *, query: str = "", fmt: str = "md") -> str:
        from ..handoff import export
        p = self.read()
        visa = self._live(p, visa)
        self._audit(visa, "handoff", {"format": fmt})
        return export(p, visa, query=query or None, fmt=fmt)

    def search(self, visa: Visa, query: str) -> dict[str, Any]:
        p = self.read()
        visa = self._live(p, visa)
        result = p.view(visa, query=query or None, budget_chars=8000)
        out = []
        for r in result.selected:
            ids = r.record.get("stamp_ids") or [r.id]
            out.append({"id": ids[0], "title": r.text.lstrip("- ")[:120], "url": self._url(ids[0])})
        self._audit(visa, "search", {"query": query[:200], "results": len(out)})
        return {"results": out}

    def fetch(self, visa: Visa, record_id: str) -> dict[str, Any]:
        p = self.read()
        visa = self._live(p, visa)
        from ..visa import scoped_copy
        copy = scoped_copy(p, visa)
        for section in _SECTIONS:
            for record in getattr(copy, section):
                if record_id in record.stamp_ids:
                    data = record.to_compact()
                    text = json.dumps({k: v for k, v in data.items() if k not in {"evidence", "source_ids"}},
                                      ensure_ascii=False, indent=2)
                    self._audit(visa, "fetch", {"id": record_id})
                    return {"id": record_id, "title": f"{section}: {data.get('key') or data.get('text', '')[:60]}",
                            "text": text, "url": self._url(record_id),
                            "metadata": {"section": section, "origin": data.get("origin", ""),
                                         "author": data.get("author", "")}}
        raise KeyError(record_id)

    def _url(self, record_id: str) -> str:
        return f"{self.base_url}/v1/records/{record_id}" if self.base_url else f"sozograph://{record_id}"

    # -- writing -------------------------------------------------------------

    def propose(self, visa: Visa, stamps: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(stamps, list) or not stamps:
            raise ValueError("stamps must be a nonempty list")
        if len(stamps) > _MAX_STAMPS:
            raise ValueError(f"at most {_MAX_STAMPS} stamps per call")
        with self.store.transaction(signer=self.signer) as p:
            live = self._live(p, visa)
            result = p.propose(stamps, visa=live, node=self.node)
        self._cache = None
        self._audit(visa, "propose", {"accepted": len(result.accepted), "quarantined": len(result.quarantined),
                                      "rejected": len(result.rejected)})
        return result.to_dict()

    def forget(self, visa: Visa, *, stamp_ids: list[str] | None = None, contains: str = "") -> dict[str, Any]:
        if not visa.forget:
            raise AccessDenied("visa lacks forget permission")
        with self.store.transaction(signer=self.signer) as p:
            live = self._live(p, visa)
            out = protocol_forget(p, stamp_ids=stamp_ids or None, contains=contains or None, visa=live,
                                  visible=lambda s: self.sees(live, s.claim))
        self._cache = None
        self._audit(visa, "forget", {"stamps": out.get("stamps", 0)})
        return out

    def sync(self, visa: Visa, stamps: list[dict[str, Any]]) -> dict[str, Any]:
        """Replica exchange for the holder's own copies. Needs a visa over every domain."""
        if "*" not in visa.domains or not (visa.write and visa.forget):
            raise AccessDenied("sync needs a replica visa (domains *, write and forget)")
        incoming = [Stamp.model_validate(s) for s in stamps]
        with self.store.transaction(signer=self.signer) as p:
            self._live(p, visa)
            p.ensure_log()
            known = {s.id for s in p.stamps}
            fresh = []
            for s in incoming:
                if s.id not in known and self.signer is not None and not s.sig:
                    s = s.model_copy(update={"sig": self.signer.sign(s)})
                fresh.append(s)
            p.stamps = union(p.stamps, fresh)
            erased = {sid for s in p.stamps if s.op == "forget" and s.admission == "accepted"
                      for sid in s.erase_sources}
            p.sources = [s for s in p.sources if s.id not in erased]
            from ..projection import project
            project(p)
            p.touch()
            out = [s.to_compact() for s in p.stamps]
        self._cache = None
        accepted = len({s.id for s in incoming} - known)
        self._audit(visa, "sync", {"received": len(incoming), "new": accepted})
        return {"stamps": out, "accepted": accepted}

    # -- holder ------------------------------------------------------------------

    def pending(self) -> list[dict[str, Any]]:
        return self.read().pending()

    def decide(self, action: str, ids: list[str]) -> dict[str, Any]:
        with self.store.transaction(signer=self.signer) as p:
            if action == "confirm":
                result = p.confirm(ids).to_dict()
            elif action == "dismiss":
                result = p.dismiss(ids).to_dict()
            elif action == "revoke":
                result = {"revoked": sum(p.revoke(i) for i in ids)}
            elif action == "revoke_all":
                result = {"revoked": p.revoke(everything=True)}
            else:
                raise ValueError(action)
        self._cache = None
        self._audit(None, action, {"ids": ids})
        return result
