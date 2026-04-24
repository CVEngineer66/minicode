from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from minicode.core.types import PermissionPolicy, ToolCapability
from minicode.features.permissions import (
    ApprovalBroker,
    DecisionStore,
    PatternSet,
    PatternRepository,
    PolicyEngine,
    ensure_tool_allowed,
    classify_dangerous_command,
    match_any_path,
    match_command,
)
from minicode.features.permissions.types import ApprovalRequest


class _MemStore:
    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def get(self, key):
        return self._d.get(key)

    def set(self, key, decision, detail=None):
        self._d[key] = decision

    def clear(self, key):
        self._d.pop(key, None)


def _make_workspace_dir() -> Path:
    base = Path.cwd() / ".codex_test_tmp"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="perm_", dir=str(base)))


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


def test_decision_store_persists_to_workspace_json() -> None:
    root = _make_workspace_dir()
    try:
        store = DecisionStore(root / ".minicode" / "permissions.json")
        store.set("write_file|path=a.txt", "allow_always", {"scope": "demo"})
        assert store.get("write_file|path=a.txt") == "allow_always"
        payload = json.loads((root / ".minicode" / "permissions.json").read_text(encoding="utf-8"))
        assert payload["decisions"]["write_file|path=a.txt"]["decision"] == "allow_always"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pattern_repository_persists_to_workspace_json() -> None:
    root = _make_workspace_dir()
    try:
        repo = PatternRepository(root / ".minicode" / "permissions.json")
        repo.save(PatternSet(allowed_commands={"git *"}, allowed_directories={str(root / "src")}))
        payload = json.loads((root / ".minicode" / "permissions.json").read_text(encoding="utf-8"))
        assert payload["patterns"]["allowedCommandPatterns"] == ["git *"]
        assert payload["patterns"]["allowedDirectoryPrefixes"] == [str(root / "src")]
        loaded = repo.load()
        assert loaded.allowed_commands == {"git *"}
        assert loaded.allowed_directories == {str(root / "src")}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_policy_engine_command_choice_uses_command_pattern() -> None:
    engine = PolicyEngine()
    choices = engine.default_choices(arguments={"command": "git status"})
    command_choice = next(choice for choice in choices if choice.get("key") == "c")
    assert command_choice["payload"]["decision"] == "allow_command_pattern"
    assert command_choice["payload"]["pattern"] == "git *"


def test_policy_engine_directory_choice_uses_directory() -> None:
    root = _make_workspace_dir()
    try:
        target = root / "src" / "main.py"
        target.parent.mkdir(parents=True)
        target.write_text("print('x')", encoding="utf-8")
        engine = PolicyEngine()
        choices = engine.default_choices(arguments={"path": "src/main.py"}, cwd=str(root))
        directory_choice = next(choice for choice in choices if choice.get("key") == "d")
        assert directory_choice["payload"]["decision"] == "allow_directory_pattern"
        assert directory_choice["payload"]["path"] == str((root / "src").resolve())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pattern_persistence_bypasses_follow_up_command_request() -> None:
    root = _make_workspace_dir()
    try:
        permissions_path = root / ".minicode" / "permissions.json"
        broker = ApprovalBroker(
            PolicyEngine(),
            DecisionStore(permissions_path),
            PatternRepository(permissions_path),
        )
        broker.add_allowed_command("git *")
        req = broker.policy_engine.build_request(
            tool_name="run_command",
            arguments={"command": "git status"},
            capability=ToolCapability(shell=True),
            policy=PermissionPolicy(kind="command"),
            mode="default",
            cwd=str(root),
        )
        assert req is None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pattern_persistence_bypasses_follow_up_directory_request() -> None:
    root = _make_workspace_dir()
    try:
        permissions_path = root / ".minicode" / "permissions.json"
        broker = ApprovalBroker(
            PolicyEngine(),
            DecisionStore(permissions_path),
            PatternRepository(permissions_path),
        )
        allowed_dir = root / "src"
        allowed_dir.mkdir()
        broker.add_allowed_directory(str(allowed_dir.resolve()))
        req = broker.policy_engine.build_request(
            tool_name="write_file",
            arguments={"path": "src/app.py"},
            capability=ToolCapability(writes_files=True),
            policy=PermissionPolicy(kind="edit"),
            mode="default",
            cwd=str(root),
        )
        assert req is None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_ensure_tool_allowed_persists_command_pattern_choice(monkeypatch) -> None:
    root = _make_workspace_dir()
    try:
        permissions_path = root / ".minicode" / "permissions.json"
        broker = ApprovalBroker(
            PolicyEngine(),
            DecisionStore(permissions_path),
            PatternRepository(permissions_path),
        )
        request = ApprovalRequest(
            kind="command",
            summary="Approve run_command",
            details=["Arguments: {'command': 'git status'}"],
            scope="run_command|command=git status",
            choices=[],
        )
        monkeypatch.setattr(
            "minicode.features.permissions.graph_adapter.interrupt",
            lambda _payload: {"decision": "allow_command_pattern", "pattern": "git *"},
        )
        decision = ensure_tool_allowed(broker, request, "run_command|command=git status")
        assert decision["decision"] == "allow_command_pattern"
        req = broker.policy_engine.build_request(
            tool_name="run_command",
            arguments={"command": "git diff"},
            capability=ToolCapability(shell=True),
            policy=PermissionPolicy(kind="command"),
            mode="default",
            cwd=str(root),
        )
        assert req is None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_match_any_path_prefix_and_glob():
    root = _make_workspace_dir()
    try:
        sub = root / "sub"
        sub.mkdir()
        file_path = sub / "a.py"
        file_path.write_text("x", encoding="utf-8")
        assert match_any_path(str(file_path), {str(root)})
        assert match_any_path(str(file_path), {"*.py"})
        assert not match_any_path(str(file_path), {"*.js"})
    finally:
        shutil.rmtree(root, ignore_errors=True)
