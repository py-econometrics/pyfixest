"""Run the Python examples in the agent skill that ships with pyfixest."""

import re
from pathlib import Path

import pytest

import pyfixest as pf

SKILL_DIR = Path(pf.__file__).parent / "skills" / "pyfixest"
SKILL_FILES = sorted(SKILL_DIR.glob("**/*.md"))
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


def test_skill_is_found():
    assert SKILL_DIR / "SKILL.md" in SKILL_FILES


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
