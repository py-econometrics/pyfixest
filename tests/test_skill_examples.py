"""Run the Python examples in the agent skill that ships with pyfixest."""

import re
from pathlib import Path

import pytest

import pyfixest as pf

SKILL_DIR = Path(pf.__file__).parent / "skills" / "pyfixest"
SKILL_FILES = sorted(SKILL_DIR.glob("**/*.md"))
INCLUDE = re.compile(r"\{\{< include (\S+) >\}\}")
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


def test_skill_is_found():
    assert SKILL_DIR / "SKILL.md" in SKILL_FILES


def test_skill_files_are_on_the_website():
    # Each file the package ships is included verbatim by a docs page, so the
    # website and llms.txt show the same text as the installed skill.
    repo_package = Path(__file__).resolve().parents[1] / "pyfixest"
    included = {
        (page.parent / target).resolve()
        for page in (repo_package.parent / "docs").rglob("*.*md")
        for target in INCLUDE.findall(page.read_text(encoding="utf-8"))
    }
    included = {
        path.relative_to(repo_package)
        for path in included
        if path.is_relative_to(repo_package)
    }
    package_dir = Path(pf.__file__).parent
    shipped = {path.relative_to(package_dir) for path in SKILL_FILES}
    assert shipped <= included, sorted(map(str, shipped - included))


def test_package_docstring_points_at_the_skill():
    skill_path = re.search(r"``(skills/\S+)``", pf.__doc__).group(1)
    assert (Path(pf.__file__).parent / skill_path).is_file()


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.filterwarnings("error::FutureWarning")
@pytest.mark.parametrize(
    "path", SKILL_FILES, ids=lambda path: str(path.relative_to(SKILL_DIR))
)
def test_skill_examples_run(path):
    # The blocks of one file share a namespace, so later blocks can reuse `fit`.
    namespace: dict = {}
    for block in PYTHON_BLOCK.findall(path.read_text(encoding="utf-8")):
        exec(compile(block, str(path), "exec"), namespace)
