"""Contract tests for the installable agent skill in `skills/pyfixest/`.

The skill file is what a coding agent loads before it writes PyFixest code, so
its frontmatter must satisfy the Agent Skills specification, it must locate the
bundled documentation independently of its install location, and its code
examples must actually run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = REPO_ROOT / "skills" / "pyfixest" / "SKILL.md"

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
