#!/usr/bin/env python3
"""Validate the required Agent Skills fields without a runtime dependency."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _frontmatter(text: str) -> dict[str, str]:
    """Read the small Agent Skills YAML frontmatter subset we require."""
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must begin with YAML frontmatter")
    _, separator, _ = text[4:].partition("\n---\n")
    if not separator:
        raise ValueError("SKILL.md frontmatter must end with a --- line")
    values: dict[str, str] = {}
    for line in text[4:].split("\n---\n", 1)[0].splitlines():
        key, colon, value = line.partition(":")
        if not colon:
            raise ValueError(f"Invalid frontmatter line: {line!r}")
        values[key.strip()] = value.strip().strip('"')
    return values


def check_skill(directory: Path) -> None:
    """Check the Agent Skills required name and description contract."""
    skill = directory / "SKILL.md"
    values = _frontmatter(skill.read_text(encoding="utf-8"))
    name, description = values.get("name", ""), values.get("description", "")
    if not NAME_RE.fullmatch(name) or len(name) > 64:
        raise ValueError(
            "Skill name must be lowercase hyphenated text up to 64 characters"
        )
    if not description or len(description) > 1_024:
        raise ValueError(
            "Skill description must be present and at most 1024 characters"
        )
    references = sorted((directory / "references").glob("*.md"))
    if len(references) != 7:
        raise ValueError("PyFixest skill must contain exactly seven focused references")
    if not (directory / "agents" / "openai.yaml").is_file():
        raise ValueError("PyFixest skill must contain generated OpenAI metadata")


def main() -> int:
    """Check one canonical skill directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        check_skill(args.directory)
    except (OSError, ValueError) as exc:
        print(f"skill validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"skill valid: {args.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
