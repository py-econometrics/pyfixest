"""Validate the canonical PyFixest skill and generate its documentation page."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

REFERENCE_NAMES = (
    "core-api.md",
    "formula-syntax.md",
    "inference.md",
    "reporting.md",
    "specialized-estimators.md",
    "demeaners.md",
    "troubleshooting.md",
)
_MARKDOWN_LINK_RE = re.compile(
    r"(?<!!)\[[^\]]+\]\((?P<target><[^>]+>|[^)\s]+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'))?\)"
)


class UserSkillError(ValueError):
    """Report invalid skill metadata, links, inventory, or generated output."""


def _frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise UserSkillError("SKILL.md must start with YAML frontmatter.")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise UserSkillError("SKILL.md frontmatter is not closed.") from exc

    values: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        if ":" not in line:
            raise UserSkillError(f"Malformed SKILL.md frontmatter line: {line}")
        key, value = line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _link_targets(text: str) -> list[str]:
    return [
        match.group("target").strip("<>") for match in _MARKDOWN_LINK_RE.finditer(text)
    ]


def _validate_links(source: Path, text: str) -> None:
    for target in _link_targets(text):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        resolved = source.parent / unquote(parsed.path)
        if not resolved.is_file():
            raise UserSkillError(f"Broken local link in {source}: {target}")


def validate_skill(skill_dir: Path) -> dict[str, str]:
    """Validate skill metadata, the seven-reference inventory, and local links."""
    main_path = skill_dir / "SKILL.md"
    try:
        main_text = main_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise UserSkillError(f"Cannot read {main_path}: {exc}") from exc

    metadata = _frontmatter(main_text)
    if metadata.get("name") != "pyfixest":
        raise UserSkillError("SKILL.md frontmatter needs name: pyfixest.")
    if not metadata.get("description"):
        raise UserSkillError("SKILL.md frontmatter needs a description.")

    references_dir = skill_dir / "references"
    actual_references = (
        {path.name for path in references_dir.glob("*.md")}
        if references_dir.is_dir()
        else set()
    )
    expected_references = set(REFERENCE_NAMES)
    if actual_references != expected_references:
        raise UserSkillError(
            "Skill reference inventory mismatch; "
            f"extra={sorted(actual_references - expected_references)}, "
            f"missing={sorted(expected_references - actual_references)}."
        )

    expected_targets = {f"references/{name}" for name in REFERENCE_NAMES}
    actual_targets = {
        target
        for target in _link_targets(main_text)
        if target.startswith("references/")
    }
    if actual_targets != expected_targets:
        raise UserSkillError("SKILL.md must link each focused reference exactly once.")

    files = {"SKILL.md": main_text}
    _validate_links(main_path, main_text)
    for name in REFERENCE_NAMES:
        path = references_dir / name
        text = path.read_text(encoding="utf-8")
        _validate_links(path, text)
        files[f"references/{name}"] = text
    return files


def _docs_page(files: dict[str, str]) -> str:
    lines = [
        "---",
        'title: "PyFixest skill for AI agents"',
        'description: "A version-controlled skill for using PyFixest reliably."',
        "---",
        "",
        "# PyFixest skill for AI agents",
        "",
        "This page is generated from the canonical files in `skills/pyfixest`.",
        "Copy the directory as a unit so its focused references remain available.",
        "",
    ]
    for relative, text in files.items():
        heading = "SKILL.md" if relative == "SKILL.md" else relative
        lines.extend(
            [
                f"## `{heading}`",
                "",
                "````markdown",
                text.rstrip(),
                "````",
                "",
            ]
        )
    return "\n".join(lines)


def sync_skill(*, skill_dir: Path, output: Path, check: bool = False) -> None:
    """Generate the public skill page, or check it for drift."""
    expected = _docs_page(validate_skill(skill_dir))
    if check:
        try:
            actual = output.read_text(encoding="utf-8")
        except OSError as exc:
            raise UserSkillError(f"Cannot read generated page {output}: {exc}") from exc
        if actual != expected:
            raise UserSkillError(f"Generated skill page is out of date: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")


def main() -> int:
    """Run skill validation and documentation synchronization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-dir", type=Path, default=Path("skills/pyfixest"))
    parser.add_argument("--output", type=Path, default=Path("docs/skills.md"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        sync_skill(skill_dir=args.skill_dir, output=args.output, check=args.check)
    except (OSError, UserSkillError) as exc:
        parser.exit(1, f"user skill check failed:\n{exc}\n")
    print("user skill check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
