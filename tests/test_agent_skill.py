"""Contract tests for the installable agent skill in `skills/pyfixest/`.

The skill file is what a coding agent loads before it writes PyFixest code, so
its frontmatter must satisfy the Agent Skills specification, its references must
stay inside the skill directory, its routing table must not drift from the cheat
sheet in either task or destination, and its code examples must actually run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = REPO_ROOT / "skills" / "pyfixest" / "SKILL.md"
CHEATSHEET_PATH = REPO_ROOT / "docs" / "cheatsheet.qmd"

MAX_BODY_LINES = 80
MAX_CODE_LINES = 20


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split a Markdown file into its YAML frontmatter and its body."""
    assert text.startswith("---\n"), "SKILL.md must open with a `---` frontmatter fence"
    closing = text.find("\n---\n", len("---\n") - 1)
    assert closing != -1, "SKILL.md frontmatter is not closed by a `---` line"
    frontmatter = text[len("---\n") : closing + 1]
    body = text[closing + len("\n---\n") :]
    return frontmatter, body


def _frontmatter_value(frontmatter: str, key: str) -> str:
    """Read a single-line `key: value` entry without a YAML dependency."""
    for line in frontmatter.splitlines():
        if line.startswith(f"{key}:"):
            return line[len(key) + 1 :].strip()
    raise AssertionError(f"SKILL.md frontmatter has no `{key}` entry")


TableRow = tuple[str, list[str], list[str]]

LINK_TARGET = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
BACKTICKED = re.compile(r"`([^`]+)`")


def _normalise_target(target: str) -> str:
    """Rewrite a cheat-sheet link target as the bundled documentation page.

    The rendered site links to `.qmd`/`.md` sources; the wheel ships the same
    pages as `.llms.md`. Any `#fragment` survives, and non-page targets such as
    `llms.txt` pass through unchanged.
    """
    path, separator, fragment = target.partition("#")
    for suffix in (".qmd", ".md"):
        if path.endswith(suffix):
            path = f"{path[: -len(suffix)]}.llms.md"
            break
    return path + separator + fragment


def _cell_targets(cell: str, *, normalise: bool) -> list[str]:
    """Return the destinations a routing-table cell points at, in order.

    The cheat sheet writes destinations as Markdown links and needs
    `normalise=True`; the skill writes them as backticked paths that are already
    bundled-documentation names.
    """
    if normalise:
        return [_normalise_target(target) for target in LINK_TARGET.findall(cell)]
    return BACKTICKED.findall(cell)


def _first_table_rows(text: str, *, normalise: bool) -> list[TableRow]:
    """Return the body rows of the first Markdown table as routing entries.

    Each entry is `(task_label, start_here_targets, then_targets)`.
    """
    rows: list[str] = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            in_table = True
            rows.append(stripped)
        elif in_table:
            break
    assert rows, "no Markdown table found"

    entries: list[TableRow] = []
    # drop the header row and the `|---|` separator
    for row in rows[2:]:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 3, f"routing-table row is not three columns: {row}"
        task, start_here, then = cells
        entries.append(
            (
                task,
                _cell_targets(start_here, normalise=normalise),
                _cell_targets(then, normalise=normalise),
            )
        )
    return entries


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def skill_body(skill_text: str) -> str:
    return _split_frontmatter(skill_text)[1]


def test_frontmatter_follows_the_agent_skills_spec(skill_text: str):
    frontmatter = _split_frontmatter(skill_text)[0]

    name = _frontmatter_value(frontmatter, "name")
    assert name == SKILL_PATH.parent.name, (
        "the skill `name` must equal its directory name so agent runtimes can "
        "resolve it"
    )

    description = _frontmatter_value(frontmatter, "description")
    assert 1 <= len(description) <= 1024
    assert "pyfixest" in description.lower()


def test_body_is_short_and_self_contained(skill_body: str):
    assert len(skill_body.strip().splitlines()) <= MAX_BODY_LINES
    # skill references resolve against the skill root, so `../` never resolves
    assert "../" not in skill_body
    # the skill must locate the bundled documentation at runtime
    assert "importlib.resources" in skill_body


def test_code_examples_run(skill_body: str):
    fences = re.findall(r"```python\n(.*?)```", skill_body, flags=re.DOTALL)
    assert fences, "SKILL.md has no ```python example"

    code = "".join(fences)
    assert len(code.strip().splitlines()) <= MAX_CODE_LINES

    compiled = compile(code, str(SKILL_PATH), "exec")
    exec(compiled, {})


def test_task_table_matches_the_cheat_sheet(skill_body: str):
    skill_rows = _first_table_rows(skill_body, normalise=False)
    cheatsheet_rows = _first_table_rows(
        CHEATSHEET_PATH.read_text(encoding="utf-8"), normalise=True
    )

    assert [row[0] for row in skill_rows] == [row[0] for row in cheatsheet_rows], (
        "the skill's routing tasks drifted from `docs/cheatsheet.qmd`"
    )
    for skill_row, cheatsheet_row in zip(skill_rows, cheatsheet_rows, strict=True):
        assert skill_row == cheatsheet_row, (
            f"the skill routes the {skill_row[0]!r} task to "
            f"{skill_row[1:]}, but `docs/cheatsheet.qmd` routes it to "
            f"{cheatsheet_row[1:]}"
        )
