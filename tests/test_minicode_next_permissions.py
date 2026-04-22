from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from minicode.core.types import PermissionPolicy, ToolCapability
from minicode.features.permissions import (
    ApprovalBroker,
    PatternSet,
    PolicyEngine,
    classify_dangerous_command,
    match_any_path,
    match_command,
)


class _MemStore:
    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def get(self, key):
        return self._d.get(key)

    def set(self, key, decision, detail=None):
        self._d[key] = decision

    def clear(self, key):
        self._d.pop(key, None)


def test_classify_dangerous_commands():
    assert classify_dangerous_command("rm", ["-rf", "/tmp/x"]) is not None
    assert classify_dangerous_command("git", ["push", "--force"]) is not None
    assert classify_dangerous_command("ls", []) is None


def test_match_command_glob():
    assert match_command("git", ["push"], {"git push"})
    assert match_command("npm", ["publish"], {"npm *"})
    assert not match_command("ls", [], {"git *"})


def test_policy_engine_hard_deny():
    engine = PolicyEngine(PatternSet(denied_commands={"git push --force"}))
    req = engine.build_request(
        tool_name="run_command",
        arguments={"command": "git", "args": ["push", "--force"]},
        capability=ToolCapability(shell=True),
        policy=PermissionPolicy(kind="command"),
        mode="default",
    )
    assert req is not None
    assert "Denied" in req.summary


def test_policy_engine_pattern_allow_bypasses_approval():
    engine = PolicyEngine(PatternSet(allowed_commands={"ls"}))
    req = engine.build_request(
        tool_name="run_command",
        arguments={"command": "ls", "args": []},
        capability=ToolCapability(shell=True),
        policy=PermissionPolicy(kind="command"),
        mode="default",
    )
    assert req is None


def test_policy_engine_plan_mode_requires_for_write():
    engine = PolicyEngine()
    req = engine.build_request(
        tool_name="write_file",
        arguments={"path": "a.txt"},
        capability=ToolCapability(writes_files=True),
        policy=PermissionPolicy(kind="edit"),
        mode="plan",
    )
    assert req is not None
    assert "Plan mode" in req.summary


def test_policy_engine_dangerous_reason_surfaces():
    engine = PolicyEngine()
    req = engine.build_request(
        tool_name="run_command",
        arguments={"command": "rm", "args": ["-rf", "/tmp/x"]},
        capability=ToolCapability(shell=True),
        policy=PermissionPolicy(kind="command"),
        mode="default",
    )
    assert req is not None
    assert any("rm -rf" in d for d in req.details)


def test_approval_broker_turn_scope():
    engine = PolicyEngine()
    broker = ApprovalBroker(engine, _MemStore())
    broker.store_decision("k", "allow_turn")
    assert broker.cached_decision("k") == "allow_turn"
    broker.begin_turn()
    assert broker.cached_decision("k") is None


def test_approval_broker_allow_all_turn():
    engine = PolicyEngine()
    broker = ApprovalBroker(engine, _MemStore())
    broker.store_decision("k", "allow_all_turn")
    assert broker.cached_decision("anything") == "allow_all_turn"


def test_approval_broker_persists_always():
    store = _MemStore()
    broker = ApprovalBroker(PolicyEngine(), store)
    broker.store_decision("k", "allow_always", {"scope": "k"})
    assert store.get("k") == "allow_always"


def test_match_any_path_prefix_and_glob(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    f = sub / "a.py"
    f.write_text("x", encoding="utf-8")
    assert match_any_path(str(f), {str(tmp_path)})
    assert match_any_path(str(f), {"*.py"})
    assert not match_any_path(str(f), {"*.js"})
