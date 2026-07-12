#!/usr/bin/env python3
"""Verify that a built PyFixest wheel contains only supported agent assets."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

FORBIDDEN_SUFFIXES = {
    ".avif",
    ".gif",
    ".ipynb",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".svg",
    ".webp",
}
SKILL_PREFIX = "skills/pyfixest/"
PACKAGE_DOCS_PREFIX = "pyfixest/docs/"


class WheelContentsError(ValueError):
    """Report a missing required asset or forbidden wheel member."""


def _validate_member_names(names: set[str]) -> list[str]:
    problems: list[str] = []
    forbidden = sorted(
        name
        for name in names
        if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES
        or "/_freeze/" in name
        or "/_site/" in name
        or name.endswith("/llms-full.txt")
    )
    if forbidden:
        problems.append(f"forbidden assets: {forbidden}")

    required = {
        f"{PACKAGE_DOCS_PREFIX}index.md",
        f"{PACKAGE_DOCS_PREFIX}llms.txt",
        f"{PACKAGE_DOCS_PREFIX}manifest.json",
        f"{SKILL_PREFIX}SKILL.md",
        f"{SKILL_PREFIX}agents/openai.yaml",
    }
    missing = sorted(required - names)
    if missing:
        problems.append(f"missing required assets: {missing}")

    reference_files = sorted(
        name
        for name in names
        if name.startswith(f"{SKILL_PREFIX}references/") and name.endswith(".md")
    )
    if len(reference_files) != 7:
        problems.append(
            "expected seven skill references under "
            f"{SKILL_PREFIX}references/, found {len(reference_files)}"
        )
    return problems


def check_wheel(path: Path) -> None:
    """Validate package docs, canonical skill, and forbidden content in one wheel."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        problems = _validate_member_names(names)
        manifest_name = f"{PACKAGE_DOCS_PREFIX}manifest.json"
        if manifest_name in names:
            manifest = json.loads(archive.read(manifest_name))
            for page in manifest.get("pages", []):
                route = page.get("route")
                expected = f"{PACKAGE_DOCS_PREFIX}pages/{route}.md"
                if not route or expected not in names:
                    problems.append(f"manifest page missing from wheel: {expected}")
        if problems:
            raise WheelContentsError(f"{path}: " + "; ".join(problems))


def _wheel_paths(inputs: list[Path]) -> list[Path]:
    """Expand wheel paths and directories without relying on a shell glob."""
    paths: list[Path] = []
    for item in inputs:
        paths.extend(sorted(item.glob("*.whl")) if item.is_dir() else [item])
    return paths


def main() -> int:
    """Check one or more wheels passed as files or directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    wheels = _wheel_paths(args.paths)
    if not wheels:
        parser.error("no wheel files found")
    for wheel in wheels:
        check_wheel(wheel)
        print(f"wheel contents valid: {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
