from __future__ import annotations

from pathlib import Path

import pytest

from scripts.sync_user_skill import REFERENCE_NAMES, UserSkillError, sync_skill


def _write_skill(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "pyfixest"
    references = skill_dir / "references"
    references.mkdir(parents=True)
    links = "\n".join(f"- [Reference](references/{name})" for name in REFERENCE_NAMES)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: pyfixest\ndescription: Test skill.\n---\n\n" + links + "\n",
        encoding="utf-8",
    )
    for name in REFERENCE_NAMES:
        (references / name).write_text(f"# {name}\n", encoding="utf-8")
    return skill_dir


def test_sync_skill_generates_and_checks_page(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path)
    output = tmp_path / "skills.md"

    sync_skill(skill_dir=skill_dir, output=output)

    page = output.read_text(encoding="utf-8")
    assert "`SKILL.md`" in page
    assert "`references/inference.md`" in page
    sync_skill(skill_dir=skill_dir, output=output, check=True)


def test_sync_skill_rejects_bad_frontmatter(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path)
    (skill_dir / "SKILL.md").write_text("# No frontmatter\n", encoding="utf-8")

    with pytest.raises(UserSkillError, match="YAML frontmatter"):
        sync_skill(skill_dir=skill_dir, output=tmp_path / "skills.md")


def test_sync_skill_rejects_broken_reference_link(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path)
    reference = skill_dir / "references/core-api.md"
    reference.write_text(
        reference.read_text(encoding="utf-8") + "[Missing](missing.md)\n",
        encoding="utf-8",
    )

    with pytest.raises(UserSkillError, match="Broken local link"):
        sync_skill(skill_dir=skill_dir, output=tmp_path / "skills.md")


def test_sync_skill_detects_generated_page_drift(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path)
    output = tmp_path / "skills.md"
    sync_skill(skill_dir=skill_dir, output=output)
    output.write_text("stale\n", encoding="utf-8")

    with pytest.raises(UserSkillError, match="out of date"):
        sync_skill(skill_dir=skill_dir, output=output, check=True)
