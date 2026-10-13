"""The MCP server: tools in process, the OAuth visa office, REST and MCP over HTTP."""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
from urllib.parse import parse_qs, urlparse

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from sozograph import Passport  # noqa: E402
from sozograph.server import build_http, build_stdio, openapi_spec  # noqa: E402

ISSUER = "http://127.0.0.1:8765"


def data_of(result):
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if structured is not None:
        return structured.get("result", structured) if isinstance(structured, dict) and len(structured) == 1 \
            and "result" in structured else structured
    return json.loads(result.content[0].text)


def call(server, name, args):
    async def main():
        async with Client(server) as client:
            return await client.call_tool(name, args)
    return anyio.run(main)


def test_stdio_server_lists_and_runs_the_passport_tools(tmp_path):
    path = tmp_path / "p.json"
    server = build_stdio(str(path), agent="claude-code", domains=["software"])

    async def main():
        async with Client(server) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert {"passport_recall", "passport_propose", "passport_history", "passport_handoff",
                    "passport_forget", "search", "fetch"} <= tools
            wrote = data_of(await client.call_tool("passport_propose", {"stamps": [
                {"attribute": "deploy_target", "value": "Cloud Run", "origin": "user_direct"},
                {"attribute": "api_key", "value": "sk-proj-" + "x" * 30}]}))
            assert len(wrote["accepted"]) == 1 and len(wrote["rejected"]) == 1
            recalled = data_of(await client.call_tool("passport_recall", {"query": "deploy"}))
            assert "Cloud Run" in recalled["context"] and recalled["agent"] == "claude-code"
            stamp = recalled["records"][0]["stamp_ids"][0]
            data_of(await client.call_tool("passport_propose", {"stamps": [
                {"op": "supersede", "target": stamp, "attribute": "deploy_target", "value": "Fly.io",
                 "origin": "user_direct"}]}))
            history = data_of(await client.call_tool("passport_history", {"attribute": "deploy_target"}))
            assert [r["status"] for r in history] == ["superseded", "current"]
            found = data_of(await client.call_tool("search", {"query": "deploy"}))
            fetched = data_of(await client.call_tool("fetch", {"id": found["results"][0]["id"]}))
            assert "Fly.io" in fetched["text"]
            packet = await client.call_tool("passport_handoff", {})
            assert "Return form" in packet.content[0].text
    anyio.run(main)
    p = Passport.load(path)
    assert p.log_backed and p.verify()["valid"]
    assert path.with_suffix(".audit.jsonl").exists()


def test_stdio_visa_is_reused_across_starts(tmp_path):
    path = str(tmp_path / "p.json")
    build_stdio(path, agent="claude-code", domains=["software"])
    build_stdio(path, agent="claude-code", domains=["software"])
    assert len(Passport.load(path).visas) == 1


def _pkce():
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def _oauth(client, *, name="Grok", domains=("marketing",), secret="s3cret-holder"):
    from starlette.testclient import TestClient  # noqa: F401
    reg = client.post("/register", json={"redirect_uris": ["http://localhost:9999/cb"], "client_name": name,
                                         "token_endpoint_auth_method": "none",
                                         "grant_types": ["authorization_code", "refresh_token"],
                                         "response_types": ["code"]})
    assert reg.status_code in (200, 201), reg.text
    client_id = reg.json()["client_id"]
    verifier, challenge = _pkce()
    auth = client.get("/authorize", params={
        "response_type": "code", "client_id": client_id, "redirect_uri": "http://localhost:9999/cb",
        "code_challenge": challenge, "code_challenge_method": "S256", "state": "xyz",
        "scope": "passport:read passport:write " + " ".join(f"domain:{d}" for d in domains),
        "resource": ISSUER + "/mcp"}, follow_redirects=False)
    assert auth.status_code in (302, 307), auth.text
    consent_url = auth.headers["location"]
    request_id = parse_qs(urlparse(consent_url).query)["request"][0]
    page = client.get("/visa-office/consent", params={"request": request_id})
    assert page.status_code == 200 and name in page.text
    wrong = client.post("/visa-office/consent", data={"request": request_id, "secret": "nope",
                                                      "decision": "approve", "domains": list(domains)})
    assert wrong.status_code == 403
    ok = client.post("/visa-office/consent", data={"request": request_id, "secret": secret, "decision": "approve",
                                                   "domains": list(domains), "write": "1", "ceiling": "1",
                                                   "days": "30"}, follow_redirects=False)
    assert ok.status_code == 302, ok.text
    redirect = parse_qs(urlparse(ok.headers["location"]).query)
    assert redirect["state"] == ["xyz"] and redirect["iss"] == [ISSUER]
    token = client.post("/token", data={"grant_type": "authorization_code", "code": redirect["code"][0],
                                        "redirect_uri": "http://localhost:9999/cb", "client_id": client_id,
                                        "code_verifier": verifier, "resource": ISSUER + "/mcp"})
    assert token.status_code == 200, token.text
    return client_id, token.json()


@pytest.fixture
def hosted(tmp_path):
    from starlette.testclient import TestClient
    app = build_http(str(tmp_path / "h.json"), issuer=ISSUER, holder_secret="s3cret-holder", server_key="k" * 64)
    with TestClient(app, base_url=ISSUER) as client:
        yield app, client, tmp_path / "h.json"


def test_discovery_metadata_points_at_the_visa_office(hosted):
    _, client, _ = hosted
    prm = client.get("/.well-known/oauth-protected-resource/mcp").json()
    assert prm["resource"].rstrip("/") == ISSUER + "/mcp" and prm["authorization_servers"][0].rstrip("/") == ISSUER
    asm = client.get("/.well-known/oauth-authorization-server").json()
    assert "S256" in asm["code_challenge_methods_supported"]
    unauthenticated = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert unauthenticated.status_code == 401
    assert "resource_metadata" in unauthenticated.headers.get("www-authenticate", "")


def test_oauth_consent_issues_a_scoped_visa_and_rest_honours_it(hosted):
    _, client, path = hosted
    _, token = _oauth(client)
    bearer = {"Authorization": f"Bearer {token['access_token']}"}
    assert "domain:marketing" in token["scope"]
    wrote = client.post("/v1/stamps", headers=bearer, json={"stamps": [
        {"attribute": "brand_voice", "value": "warm, spare, declarative", "origin": "user_direct"}]}).json()
    assert len(wrote["accepted"]) == 1
    outside = client.post("/v1/stamps", headers=bearer, json={"stamps": [
        {"attribute": "editor", "value": "vim", "domains": ["software"]}]}).json()
    assert outside["rejected"]
    recalled = client.get("/v1/recall", headers=bearer, params={"query": "voice"}).json()
    assert "declarative" in recalled["context"]
    assert client.get("/v1/recall").status_code == 401
    p = Passport.load(path)
    assert p.visas[0].agent == "grok" and p.verify()["valid"]
    assert all(s.sig.startswith("hmac-sha256:server:") for s in p.stamps)
    assert token["access_token"] not in path.read_text(encoding="utf-8")


def test_refresh_rotates_the_visa_token_and_revocation_ends_it(hosted):
    _, client, path = hosted
    client_id, token = _oauth(client)
    rotated = client.post("/token", data={"grant_type": "refresh_token", "refresh_token": token["refresh_token"],
                                          "client_id": client_id, "resource": ISSUER + "/mcp"})
    assert rotated.status_code == 200, rotated.text
    new = rotated.json()["access_token"]
    assert client.get("/v1/recall", headers={"Authorization": f"Bearer {token['access_token']}"}).status_code == 401
    assert client.get("/v1/recall", headers={"Authorization": f"Bearer {new}"}).status_code == 200
    login = client.post("/visa-office/login", data={"secret": "s3cret-holder"}, follow_redirects=False)
    cookie = login.cookies.get("sozograph_holder") or login.headers["set-cookie"].split(";")[0].split("=", 1)[1]
    client.cookies.set("sozograph_holder", cookie)
    home = client.get("/visa-office")
    assert "grok" in home.text and "Revoke" in home.text
    csrf = home.text.split("name=csrf value=")[1].split(">")[0]
    client.post("/visa-office/action", data={"csrf": csrf, "action": "revoke_all", "ids": "all"},
                follow_redirects=False)
    assert client.get("/v1/recall", headers={"Authorization": f"Bearer {new}"}).status_code == 401


def test_holder_confirms_quarantine_in_the_visa_office(hosted):
    _, client, path = hosted
    _, token = _oauth(client, name="Muse", domains=("shopping",))
    bearer = {"Authorization": f"Bearer {token['access_token']}"}
    out = client.post("/v1/stamps", headers=bearer, json={"stamps": [
        {"attribute": "shipping_address", "value": "12 Baines Ave", "origin": "third_party"}]}).json()
    assert out["quarantined"]
    login = client.post("/visa-office/login", data={"secret": "s3cret-holder"}, follow_redirects=False)
    client.cookies.set("sozograph_holder", login.cookies.get("sozograph_holder"))
    home = client.get("/visa-office")
    assert "12 Baines Ave" in home.text
    csrf = home.text.split("name=csrf value=")[1].split(">")[0]
    client.post("/visa-office/action", data={"csrf": csrf, "action": "confirm", "ids": out["quarantined"][0]["id"]},
                follow_redirects=False)
    assert any(f.key == "shipping_address" for f in Passport.load(path).facts)


def test_mcp_over_streamable_http_with_a_visa_token(hosted):
    import httpx2
    from mcp.client.streamable_http import streamable_http_client
    _, client, path = hosted
    _, token = _oauth(client)
    # A fresh process over the same files: visas and office state persist on disk.
    app = build_http(str(path), issuer=ISSUER, holder_secret="s3cret-holder", server_key="k" * 64)

    async def main():
        async with app.router.lifespan_context(app):
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=ISSUER,
                                          headers={"Authorization": f"Bearer {token['access_token']}"}) as http:
                async with Client(streamable_http_client(ISSUER + "/mcp", http_client=http)) as mcp_client:
                    wrote = data_of(await mcp_client.call_tool("passport_propose", {"stamps": [
                        {"attribute": "campaign", "value": "October launch", "origin": "user_direct"}]}))
                    assert wrote["accepted"]
                    recalled = data_of(await mcp_client.call_tool("passport_recall", {"query": "launch"}))
                    assert "October launch" in recalled["context"] and recalled["agent"] == "grok"
    anyio.run(main)


def test_sync_needs_a_replica_visa_and_merges_logs(hosted, tmp_path):
    _, client, path = hosted
    p = Passport.load(path)
    replica = p.issue_visa("laptop", domains=["*"], write=True, forget=True)
    weak = p.issue_visa("weak", domains=["shopping"], write=True)
    p.save(path)
    local = Passport.issue()
    local.propose([{"attribute": "editor", "value": "VS Code"}])
    body = {"stamps": [s.to_compact() for s in local.stamps]}
    assert client.post("/v1/sync", headers={"Authorization": f"Bearer {weak.token}"}, json=body).status_code == 403
    out = client.post("/v1/sync", headers={"Authorization": f"Bearer {replica.token}"}, json=body).json()
    assert out["accepted"] == 1
    hosted_p = Passport.load(path)
    assert any(f.key == "editor" for f in hosted_p.facts)


def test_openapi_describes_the_rest_mirror():
    spec = openapi_spec("https://passport.example.com")
    assert spec["servers"][0]["url"] == "https://passport.example.com"
    assert {"/v1/recall", "/v1/stamps", "/v1/history", "/v1/handoff", "/v1/forget"} <= set(spec["paths"])
