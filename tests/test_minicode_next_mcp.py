from __future__ import annotations

import pytest

from minicode.features.mcp import (
    McpClientPool,
    McpConnectionError,
    McpService,
    McpValidationError,
    sanitize_tool_segment,
    validate_args,
    validate_command,
)


class _FakeRepo:
    def __init__(self, servers):
        self.servers = list(servers)

    def list(self):
        return self.servers

    def add(self, name, command, args, env, cwd):
        self.servers.append(
            {"name": name, "command": command, "args": args, "env": env, "cwd": cwd}
        )


class _FakeClient:
    def __init__(self, *, alive=True, tools=None, fail=None):
        self._alive = alive
        self._tools = tools or [{"name": "demo"}]
        self._fail = fail
        self.close_called = False

    def is_alive(self):
        return self._alive

    def list_tools(self):
        if self._fail == "tools":
            raise McpConnectionError("boom")
        return self._tools

    def call_tool(self, name, arguments):
        if self._fail == "call":
            raise McpConnectionError("boom")
        return {"name": name, "args": arguments}

    def close(self):
        self.close_called = True
        self._alive = False


def test_sanitize_tool_segment():
    assert sanitize_tool_segment("weird name!") == "weird_name"
    assert sanitize_tool_segment("") == "tool"
    assert sanitize_tool_segment("___") == "tool"


def test_validate_args_rejects_shell_chars():
    validate_args(["--safe", "path"])
    with pytest.raises(McpValidationError):
        validate_args(["rm ; echo x"])


def test_validate_command_rejects_shells():
    with pytest.raises(McpValidationError):
        validate_command("bash")


def test_validate_command_allows_whitelisted():
    validate_command("python")
    validate_command("node")


def test_pool_get_reuses_alive_client():
    calls = {"n": 0}

    def factory(name: str):
        calls["n"] += 1
        return _FakeClient()

    pool = McpClientPool(factory=factory)
    c1 = pool.get("x")
    c2 = pool.get("x")
    assert c1 is c2
    assert calls["n"] == 1


def test_pool_respawns_on_dead_client():
    count = {"n": 0}
    clients: list[_FakeClient] = []

    def factory(name: str):
        count["n"] += 1
        c = _FakeClient()
        clients.append(c)
        return c

    pool = McpClientPool(factory=factory)
    pool.get("x")
    clients[0]._alive = False
    pool.get("x")
    assert count["n"] == 2
    assert clients[0].close_called


def test_pool_health_check_drops_on_failure():
    def factory(name: str):
        return _FakeClient(fail="tools")

    pool = McpClientPool(factory=factory)
    assert pool.health_check("x") is False
    assert "x" not in pool.stats()["active"]


def test_service_call_tool_success(monkeypatch):
    svc = McpService(_FakeRepo([{"name": "s", "command": "node", "args": [], "env": {}, "cwd": None}]))
    monkeypatch.setattr(svc, "_spawn_client", lambda name: _FakeClient())
    out = svc.call_tool("s", "demo", {"x": 1})
    assert out["ok"] is True
    assert out["result"]["args"] == {"x": 1}


def test_service_call_tool_failure_returns_structured_error(monkeypatch):
    svc = McpService(_FakeRepo([{"name": "s", "command": "node", "args": [], "env": {}, "cwd": None}]))
    monkeypatch.setattr(svc, "_spawn_client", lambda name: _FakeClient(fail="call"))
    out = svc.call_tool("s", "demo", {})
    assert out["ok"] is False
    assert "connection" in out["error"]


def test_service_list_tools_cached(monkeypatch):
    svc = McpService(
        _FakeRepo([{"name": "s", "command": "node", "args": [], "env": {}, "cwd": None}]),
        tool_cache_ttl_s=10,
    )
    calls = {"n": 0}

    def factory(name: str):
        calls["n"] += 1
        return _FakeClient()

    monkeypatch.setattr(svc, "_spawn_client", factory)
    svc.list_tools("s")
    svc.list_tools("s")
    assert calls["n"] == 1  # second call hits cache
