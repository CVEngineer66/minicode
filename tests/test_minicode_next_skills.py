from __future__ import annotations

import pytest

from minicode.features.skills import (
    SkillConflictError,
    SkillService,
    content_hash,
    extract_description,
    extract_metadata,
    parse_frontmatter,
)


class _FakeRepo:
    def __init__(self):
        self.store: dict[str, dict] = {}
        self.skills_dir = None

    def install(self, name, source, body):
        self.store[name] = {"name": name, "source": source, "body": body, "installed_at": 0}

    def list(self):
        return list(self.store.values())

    def get(self, name):
        return self.store.get(name)


SAMPLE_BODY = """---
version: 1.2.0
description: Formal description from frontmatter
---

# Title

A short skill that does stuff.
"""


def test_parse_frontmatter_and_description():
    fm = parse_frontmatter(SAMPLE_BODY)
    assert fm["version"] == "1.2.0"
    # Description falls back to first non-heading if no frontmatter
    plain = "# H\n\nFirst line here."
    assert "First line here" in extract_description(plain)


def test_extract_metadata_fields():
    meta = extract_metadata("demo", SAMPLE_BODY, source="user")
    assert meta.version == "1.2.0"
    assert meta.description.startswith("Formal description")
    assert len(meta.content_hash) == 16


def test_content_hash_stable():
    h1 = content_hash("abc")
    h2 = content_hash("abc")
    h3 = content_hash("abcd")
    assert h1 == h2
    assert h1 != h3


def test_install_skill_trusted_hashes():
    body = "hello"
    svc = SkillService(_FakeRepo(), trusted_hashes={content_hash(body)})
    meta = svc.install_skill("demo", body, source="user")
    assert meta.content_hash == content_hash(body)


def test_install_skill_untrusted_rejected():
    svc = SkillService(_FakeRepo(), trusted_hashes={"deadbeef" * 2})
    with pytest.raises(SkillConflictError):
        svc.install_skill("demo", "untrusted body", source="user")


def test_install_skill_conflict_without_overwrite():
    svc = SkillService(_FakeRepo())
    svc.install_skill("demo", "v1", source="user")
    with pytest.raises(SkillConflictError):
        svc.install_skill("demo", "v2", source="user")


def test_install_skill_overwrite_allowed():
    svc = SkillService(_FakeRepo())
    svc.install_skill("demo", "v1", source="user")
    meta = svc.install_skill("demo", "v2", source="user", overwrite=True)
    assert meta.content_hash == content_hash("v2")


def test_discover_deduplicates_by_name(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    for root in (root_a, root_b):
        (root / "demo").mkdir(parents=True)
        (root / "demo" / "SKILL.md").write_text(
            f"---\nversion: 1\n---\n\nDemo in {root.name}.", encoding="utf-8"
        )
    svc = SkillService(_FakeRepo(), discovery_roots=[(root_a, "project"), (root_b, "user")])
    found = svc.discover()
    # Only the first (project) root wins for name "demo"
    assert len(found) == 1
    assert found[0].source == "project"


def test_install_from_path(tmp_path):
    skill_dir = tmp_path / "hello"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# hello\n\nDoes stuff.\n", encoding="utf-8")
    svc = SkillService(_FakeRepo())
    meta = svc.install_from_path(skill_dir)
    assert meta.name == "hello"
    assert "Does stuff" in meta.description


def test_check_conflict():
    svc = SkillService(_FakeRepo())
    svc.install_skill("demo", "v1", source="u")
    assert svc.check_conflict("demo", "v1") is None
    assert svc.check_conflict("demo", "v2") is not None
    assert svc.check_conflict("new", "x") is None
