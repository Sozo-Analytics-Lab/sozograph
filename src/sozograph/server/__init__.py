"""
SozoGraph MCP server (optional extra: `pip install "sozograph[mcp]"`).

    sozograph serve --stdio --passport holder.json --agent claude-code --domains software
    sozograph serve --http --passport holder.json --issuer https://passport.example.com
"""
from __future__ import annotations

try:
    import mcp  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError("The SozoGraph server needs the MCP SDK") from exc

from .app import build_http, build_mcp, build_stdio, openapi_spec, run
from .service import AccessDenied, PassportService

__all__ = ["build_http", "build_mcp", "build_stdio", "openapi_spec", "run", "PassportService", "AccessDenied"]
