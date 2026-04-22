from __future__ import annotations

import time
from typing import Any

from .client import McpClient, McpConnectionError, McpTimeoutError
from .pool import McpClientPool


class McpService:
    """Cached MCP access with graceful failure degradation.

    Boundaries:
    - Tool lists are cached per server for `tool_cache_ttl_s` (default 60s).
    - Failures return structured `{"ok": False, "error": str}` instead of raising.
    """

    def __init__(
        self,
        repository: Any,
        *,
        connect_timeout_s: float = 5.0,
        call_timeout_s: float = 60.0,
        tool_cache_ttl_s: float = 60.0,
    ) -> None:
        self.repository = repository
        self.connect_timeout_s = connect_timeout_s
        self.call_timeout_s = call_timeout_s
        self.tool_cache_ttl_s = tool_cache_ttl_s
        self._tool_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self.pool = McpClientPool(factory=self._spawn_client)

    def add_server(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str],
        cwd: str | None = None,
    ) -> None:
        self.repository.add(name, command, args, env, cwd)
        self._tool_cache.pop(name, None)
        self.pool.drop(name)

    def list_servers(self) -> list[dict[str, Any]]:
        return self.repository.list()

    def list_tools(self, name: str, *, use_cache: bool = True) -> list[dict[str, Any]]:
        if use_cache:
            cached = self._tool_cache.get(name)
            if cached and (time.time() - cached[0]) < self.tool_cache_ttl_s:
                return cached[1]
        try:
            tools = self.pool.get(name).list_tools()
        except (McpConnectionError, McpTimeoutError):
            return []
        self._tool_cache[name] = (time.time(), tools)
        return tools

    def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            result = self.pool.get(server_name).call_tool(tool_name, arguments)
            return {"ok": True, "result": result}
        except McpTimeoutError as exc:
            self.pool.drop(server_name)
            return {"ok": False, "error": f"timeout: {exc}"}
        except McpConnectionError as exc:
            self.pool.drop(server_name)
            self._tool_cache.pop(server_name, None)
            return {"ok": False, "error": f"connection: {exc}"}
        except BaseException as exc:  # defensive
            return {"ok": False, "error": f"mcp: {exc}"}

    def health_check(self, name: str) -> bool:
        ok = self.pool.health_check(name)
        if not ok:
            self._tool_cache.pop(name, None)
        return ok

    def reap_idle(self) -> list[str]:
        return self.pool.reap_idle()

    def shutdown(self) -> None:
        self.pool.close_all()
        self._tool_cache.clear()

    # --- internals ---
    def _spawn_client(self, name: str) -> McpClient:
        server = next((s for s in self.repository.list() if s["name"] == name), None)
        if server is None:
            raise McpConnectionError(f"Unknown MCP server: {name}")
        return McpClient(
            server["command"],
            server["args"],
            env=server["env"],
            cwd=server["cwd"],
            connect_timeout_s=self.connect_timeout_s,
            call_timeout_s=self.call_timeout_s,
        )
