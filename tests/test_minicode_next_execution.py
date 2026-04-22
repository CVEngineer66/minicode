from __future__ import annotations

from pathlib import Path

import pytest

from minicode.features.execution import (
    Decision,
    ExecutionService,
    PathEscapeError,
    RiskLevel,
    assess_command_risk,
    is_sensitive_path,
    resolve_path_within,
)


def test_assess_command_risk_critical():
    assert assess_command_risk("rm", ["-rf", "/"]) == RiskLevel.CRITICAL
    assert assess_command_risk("dropdb", []) == RiskLevel.CRITICAL


def test_assess_command_risk_safe_and_medium():
    assert assess_command_risk("ls", []) == RiskLevel.SAFE
    assert assess_command_risk("git", ["status"]) == RiskLevel.MEDIUM
    assert assess_command_risk("unknown-binary", []) == RiskLevel.MEDIUM


def test_assess_command_risk_destructive_flag():
    assert assess_command_risk("some-tool", ["--force"]) == RiskLevel.CRITICAL


def test_resolve_path_within_accepts_inside(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    target = tmp_path / "sub" / "a.txt"
    target.write_text("x", encoding="utf-8")
    resolved = resolve_path_within(target, [tmp_path])
    assert resolved == target.resolve()


def test_resolve_path_within_rejects_escape(tmp_path: Path):
    with pytest.raises(PathEscapeError):
        resolve_path_within("../etc/passwd", [tmp_path])


def test_is_sensitive_path():
    assert is_sensitive_path(".env")
    assert is_sensitive_path("/home/u/.ssh/id_rsa")
    assert is_sensitive_path("certs/server.pem")
    assert not is_sensitive_path("src/app.py")


def test_service_check_path_access_deny(tmp_path: Path):
    svc = ExecutionService(allowed_roots=[tmp_path])
    decision = svc.check_path_access("/outside/file.txt", write=True)
    assert decision.decision == Decision.DENY


def test_service_check_path_access_sensitive(tmp_path: Path):
    svc = ExecutionService(allowed_roots=[tmp_path])
    target = tmp_path / ".env"
    target.write_text("SECRET=1", encoding="utf-8")
    decision = svc.check_path_access(target, write=True)
    assert decision.decision == Decision.REQUIRE_APPROVAL
    assert decision.risk == RiskLevel.HIGH


def test_service_check_command_maps_risk():
    svc = ExecutionService()
    assert svc.check_command("ls").decision == Decision.ALLOW
    assert svc.check_command("git", ["status"]).decision == Decision.ALLOW
    assert svc.check_command("sudo", ["apt-get", "update"]).decision == Decision.REQUIRE_APPROVAL
    assert svc.check_command("rm", ["-rf", "x"]).decision == Decision.REQUIRE_APPROVAL


def test_service_no_whitelist_allows_all(tmp_path: Path):
    svc = ExecutionService(allowed_roots=[])
    assert svc.check_path_access("/outside").decision == Decision.ALLOW


def test_format_risk_info():
    svc = ExecutionService()
    text = svc.format_risk_info("git", ["push"])
    assert "Risk" in text
    assert "MEDIUM" in text
