"""Classify a pyfixest diff for risk-based verification."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


class ScopeError(RuntimeError):
    """Raised when the change scope cannot be determined."""


@dataclass(frozen=True, slots=True)
class ChangeScope:
    """Classified repository changes.

    Attributes
    ----------
    base
        Git revision used to calculate the merge base.
    merge_base
        Resolved merge-base commit.
    files
        Sorted changed paths relative to the repository root.
    domains
        Sorted verification domains selected by the changed paths.
    risks
        Sorted high-risk behaviors implicated by the changed paths.
    """

    base: str
    merge_base: str
    files: tuple[str, ...]
    domains: tuple[str, ...]
    risks: tuple[str, ...]

    def to_dict(self) -> dict[str, str | list[str]]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["files"] = list(self.files)
        payload["domains"] = list(self.domains)
        payload["risks"] = list(self.risks)
        return payload


def _git(args: Sequence[str], cwd: Path) -> str:
    """Run a read-only Git command and return stripped stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ScopeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def changed_files(base: str, cwd: Path) -> tuple[str, tuple[str, ...]]:
    """Resolve the merge base and return committed and local changed paths."""
    merge_base = _git(["merge-base", "HEAD", base], cwd=cwd)
    tracked = _git(
        ["diff", "--name-only", "--diff-filter=ACMRD", merge_base],
        cwd=cwd,
    ).splitlines()
    untracked = _git(
        ["ls-files", "--others", "--exclude-standard"],
        cwd=cwd,
    ).splitlines()
    files = tuple(sorted({path for path in [*tracked, *untracked] if path}))
    return merge_base, files


def classify_path(path: str) -> set[str]:
    """Map one repository path to verification domains."""
    domains: set[str] = set()

    if path.startswith(".agents/") or path.startswith("scripts/agent/"):
        domains.add("agent")
    if path.startswith(".github/"):
        domains.add("ci")
    if path.startswith("docs/") or path in {"CONTRIBUTING.md", "README.md"}:
        domains.add("docs")
    if path.startswith("scripts/reference/"):
        domains.add("reference")
    if path.startswith("tests/"):
        domains.add("tests")
        if "hac" in path:
            domains.add("hac")
        if "fixest" in path or path.startswith("tests/data/"):
            domains.add("reference")
        if "numba" in path or "nojit" in path:
            domains.add("numba")
    if path.startswith("src/") or path.startswith("pyfixest/core/"):
        domains.add("rust")
    if path.startswith("pyfixest/estimation/api/") or path == "pyfixest/__init__.py":
        domains.update({"api", "numerical", "python"})
    elif path.startswith("pyfixest/estimation/") or path.startswith("pyfixest/did/"):
        domains.update({"numerical", "python"})
    elif path.startswith("pyfixest/") or path.endswith((".py", ".pyi")):
        domains.add("python")
    if path in {"pyproject.toml", "pixi.lock", "Cargo.toml", "Cargo.lock"}:
        domains.add("packaging")
        if path in {"pixi.lock", "Cargo.lock"}:
            domains.add("dependencies")

    return domains or {"unknown"}


def risk_flags(path: str) -> set[str]:
    """Return high-risk behavior flags implied by one path."""
    risks: set[str] = set()

    if path.startswith("pyfixest/estimation/api/") or path == "pyfixest/__init__.py":
        risks.add("public-api")
    if path.startswith(
        (
            "pyfixest/estimation/models/",
            "pyfixest/estimation/internals/",
            "pyfixest/estimation/formula/",
        )
    ) or path in {
        "pyfixest/estimation/config.py",
        "pyfixest/estimation/plan_.py",
        "pyfixest/estimation/runner.py",
    }:
        risks.add("shared-estimation-core")
    if "formula" in path:
        risks.add("formula-semantics")
    if "vcov" in path or "inference" in path or "ses" in path:
        risks.add("inference")
    if path.startswith("src/") or path.startswith("pyfixest/core/"):
        risks.add("native-kernel")
    if path.startswith("tests/data/") or path.startswith("scripts/reference/"):
        risks.add("stored-reference")
    if path in {"pyproject.toml", "pixi.lock", "Cargo.toml", "Cargo.lock"}:
        risks.add("dependency-or-build")

    return risks


def classify_changes(
    *,
    base: str,
    merge_base: str,
    files: Sequence[str],
) -> ChangeScope:
    """Build a deterministic scope from changed paths."""
    domains: set[str] = set()
    risks: set[str] = set()
    for path in files:
        domains.update(classify_path(path))
        risks.update(risk_flags(path))
    return ChangeScope(
        base=base,
        merge_base=merge_base,
        files=tuple(sorted(set(files))),
        domains=tuple(sorted(domains)),
        risks=tuple(sorted(risks)),
    )


def get_change_scope(base: str, cwd: Path) -> ChangeScope:
    """Inspect Git and return the classified change scope."""
    merge_base, files = changed_files(base=base, cwd=cwd)
    return classify_changes(base=base, merge_base=merge_base, files=files)


def _format_scope(scope: ChangeScope) -> str:
    """Format a scope for terminal review."""
    lines = [
        f"Base: {scope.base}",
        f"Merge base: {scope.merge_base}",
        f"Domains: {', '.join(scope.domains) or 'none'}",
        f"Risks: {', '.join(scope.risks) or 'none'}",
        "Files:",
    ]
    lines.extend(f"  - {path}" for path in scope.files)
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="origin/master",
        help="Git revision used to calculate the merge base.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the human-readable summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scope-classification CLI."""
    args = _parse_args(argv)
    try:
        scope = get_change_scope(base=args.base, cwd=Path.cwd())
    except ScopeError as exc:
        print(f"error: {exc}")
        return 2

    if args.json:
        print(json.dumps(scope.to_dict(), indent=2, sort_keys=True))
    else:
        print(_format_scope(scope))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
