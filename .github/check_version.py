"""Validate that the git tag version matches Cargo.toml version.

Adapted from pydantic-core's version check script.
See: https://github.com/pydantic/pydantic-core/blob/main/.github/check_version.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def get_cargo_version() -> str:
    """Read the version string from Cargo.toml."""
    cargo_toml = Path(__file__).resolve().parent.parent / "Cargo.toml"
    content = cargo_toml.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        print("ERROR: Could not find version in Cargo.toml")
        sys.exit(1)
    return match.group(1)


def cargo_to_python_version(cargo_version: str) -> str:
    """Convert Cargo SemVer pre-release to PEP 440.

    Examples
    --------
        0.50.0 -> 0.50.0
        0.50.0-alpha.1 -> 0.50.0a1
        0.50.0-beta.1 -> 0.50.0b1
    """
    if "-alpha." in cargo_version:
        base, num = cargo_version.split("-alpha.")
        return f"{base}a{num}"
    if "-beta." in cargo_version:
        base, num = cargo_version.split("-beta.")
        return f"{base}b{num}"
    return cargo_version


def is_prerelease_version(cargo_version: str) -> bool:
    """Return whether a Cargo SemVer version has a prerelease component."""
    version_without_build_metadata = cargo_version.split("+", maxsplit=1)[0]
    return "-" in version_without_build_metadata


def main() -> None:
    """Check that the git tag matches the Cargo.toml version."""
    import os

    github_ref = os.environ.get("GITHUB_REF", "")
    if not github_ref.startswith("refs/tags/"):
        print(f"Not a tag ref: {github_ref}")
        sys.exit(1)

    tag = github_ref.removeprefix("refs/tags/")
    tag_version = tag.removeprefix("v")

    cargo_version = get_cargo_version()
    python_version = cargo_to_python_version(cargo_version)

    if tag_version != python_version:
        print(f"MISMATCH: tag={tag_version!r} != Cargo.toml={python_version!r}")
        sys.exit(1)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        prerelease = str(is_prerelease_version(cargo_version)).lower()
        with Path(github_output).open("a", encoding="utf-8") as output_file:
            output_file.write(f"is_prerelease={prerelease}\n")

    print(f"OK: tag={tag_version!r} matches Cargo.toml={python_version!r}")


if __name__ == "__main__":
    main()
