"""
Mechanical house-style checks that back the prose rules in `AGENTS.md`.

Every check here exists because a real PR broke the rule. Rules that can live
in this file do not also live in `AGENTS.md` — the guide states intent, this
script enforces it. Add a check whenever a review finding turns out to be
mechanically detectable, and delete the corresponding prose.

Two modes:

    python scripts/check_house_style.py [FILE ...]   # tree checks (pre-commit)
    python scripts/check_house_style.py --diff BASE  # diff checks (CI)

Exits non-zero if anything is found.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

Finding = tuple[str, int, str]

# --- tree checks -----------------------------------------------------------

NUMBA_ALLOWED = (
    "pyfixest/estimation/numba/",
    "pyfixest/estimation/post_estimation/ritest.py",
)
GENERATED = (
    "pixi.lock",
    "Cargo.lock",
    "docs/_freeze/",
    "coverage.xml",
    "docs/reference/",
)

CAMEL_DEFAULT = re.compile(r'=\s*"([a-z]+[A-Z][a-zA-Z]*)"')

# Files that predate a rule. Do not extend this list — new code follows the
# rule, and an entry leaves only when the file is cleaned up.
GRANDFATHERED = {
    "tests/test_check_version.py": {"test class"},
    "tests/test_formula_parse.py": {"test class"},
    "tests/test_hac_meat.py": {"test class"},
    "pyfixest/estimation/deprecated/FormulaParser.py": {"doctest example"},
}


def check_file(path: str, text: str) -> list[Finding]:
    """Run every tree check that applies to one file."""
    out: list[Finding] = []
    lines = text.splitlines()
    in_tests = path.startswith("tests/")
    in_pkg = path.startswith("pyfixest/")

    for i, line in enumerate(lines, start=1):
        if in_tests and re.match(r"\s*class Test", line):
            out.append((path, i, "test class — use module-level test_* functions"))
        if in_tests and "def setup_method" in line:
            out.append((path, i, "setup_method — use a seeded fixture instead"))
        if in_tests and ("spec_from_file_location" in line or "importlib.util" in line):
            out.append((path, i, "importlib file loading — import through the package"))
        if in_pkg and line.lstrip().startswith(">>>"):
            out.append((path, i, "doctest example — use an executable {python} chunk"))
        if (
            in_pkg
            and re.match(r"\s*(import numba|from numba)", line)
            and not any(path.startswith(a) for a in NUMBA_ALLOWED)
        ):
            out.append(
                (path, i, "new numba path — hot loops become Rust kernels in src/")
            )
        if path.startswith("pyfixest/estimation/api/") and "data: pd.DataFrame" in line:
            out.append(
                (path, i, "pd.DataFrame in an api/ signature — use DataFrameType")
            )
        for m in CAMEL_DEFAULT.finditer(line):
            out.append(
                (path, i, f'camelCase string default "{m.group(1)}" — use snake_case')
            )

    exempt = GRANDFATHERED.get(path, set())
    return [f for f in out if not any(f[2].startswith(e) for e in exempt)]


def check_rpy2_registration(paths: list[str]) -> list[Finding]:
    """Flag test files that import rpy2 without a `_rpy2_test_files` entry."""
    conftest = ROOT / "tests" / "conftest.py"
    if not conftest.exists():
        return []
    registered = conftest.read_text()
    out: list[Finding] = []
    for path in paths:
        if not (path.startswith("tests/") and path.endswith(".py")):
            continue
        if Path(path).name == "conftest.py":
            continue
        full = ROOT / path
        if not full.exists() or "rpy2" not in full.read_text():
            continue
        if Path(path).name not in registered:
            out.append((path, 1, "imports rpy2 but is missing from _rpy2_test_files"))
    return out


# --- diff checks -----------------------------------------------------------


def _diff(base: str, *args: str) -> str:
    cmd = ["git", "diff", base, *args]
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=ROOT, check=False
    ).stdout


def check_deleted_tests(base: str) -> list[Finding]:
    """Flag removed tests — tests are added, never replaced."""
    out: list[Finding] = []
    for line in _diff(base, "--", "tests/").splitlines():
        m = re.match(r"^-def (test_\w+)", line)
        if m:
            out.append(
                (
                    "tests/",
                    0,
                    f"deleted test `{m.group(1)}` — tests are added, not replaced",
                )
            )
    return out


def check_generated_churn(base: str) -> list[Finding]:
    """Flag generated artifacts and lockfiles left in a feature diff."""
    out: list[Finding] = []
    for stat in _diff(base, "--numstat").splitlines():
        fields = stat.split("\t")
        if len(fields) != 3:
            continue
        added, _removed, name = fields
        if not any(name.startswith(g) or name == g for g in GENERATED):
            continue
        # A lockfile edit under ~5 lines is the local-package hash bump that any
        # pyproject.toml change produces; a re-resolve is orders of magnitude
        # larger and is what this check is actually looking for.
        if name.endswith(".lock") and added.isdigit() and int(added) <= 5:
            continue
        out.append(
            (
                name,
                0,
                "generated/lockfile churn — drop it unless the task is about dependencies",
            )
        )
    return out


def check_data_provenance(base: str) -> list[Finding]:
    """Every tests/data/ file ships with the script that generated it."""
    names = [
        n
        for n in _diff(base, "--name-only").splitlines()
        if n.startswith("tests/data/")
    ]
    added = [n for n in names if not n.endswith((".do", ".R", ".py"))]
    has_generator = any(n.endswith((".do", ".R", ".py")) for n in names)
    if added and not has_generator:
        return [
            (added[0], 0, "new tests/data/ file with no generator script in the diff")
        ]
    return []


def check_quartodoc(base: str) -> list[Finding]:
    """Flag new public model methods missing from the quartodoc reference."""
    quarto = (ROOT / "docs" / "_quarto.yml").read_text()
    out: list[Finding] = []
    for line in _diff(base, "--", "pyfixest/estimation/models/").splitlines():
        m = re.match(r"^\+    def ([a-z]\w+)\(", line)
        if m and m.group(1) not in quarto:
            out.append(
                (
                    "docs/_quarto.yml",
                    0,
                    f"new public method `{m.group(1)}` missing from quartodoc contents",
                )
            )
    return out


# --- entry point -----------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files", nargs="*", help="files to check (default: all tracked)"
    )
    parser.add_argument("--diff", metavar="BASE", help="run diff checks against BASE")
    args = parser.parse_args()

    findings: list[Finding] = []

    if args.diff:
        findings += check_deleted_tests(args.diff)
        findings += check_generated_churn(args.diff)
        findings += check_data_provenance(args.diff)
        findings += check_quartodoc(args.diff)
    else:
        paths = (
            args.files
            or subprocess.run(
                ["git", "ls-files", "*.py"],
                capture_output=True,
                text=True,
                cwd=ROOT,
                check=False,
            ).stdout.split()
        )
        for path in paths:
            rel = (
                str(Path(path).resolve().relative_to(ROOT))
                if Path(path).is_absolute()
                else path
            )
            full = ROOT / rel
            if full.exists() and full.suffix == ".py":
                findings += check_file(rel, full.read_text())
        findings += check_rpy2_registration([str(p) for p in paths])

    for path, line, msg in findings:
        location = f"{path}:{line}" if line else path
        print(f"{location}: {msg}")

    if findings:
        print(f"\n{len(findings)} house-style finding(s) — see AGENTS.md")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
