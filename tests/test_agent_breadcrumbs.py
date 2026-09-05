"""Check the breadcrumbs an agent reads out of `help(pyfixest)`.

An agent that only has the installed package points itself at the documentation
and the skill through the package docstring, so the three breadcrumb lines are a
contract rather than prose.
"""

from __future__ import annotations

import pyfixest as pf


def test_docstring_points_at_the_bundled_documentation():
    assert pf.__doc__ is not None
    assert 'importlib.resources.files("pyfixest") / "docs"' in pf.__doc__
    assert "https://pyfixest.org/llms.txt" in pf.__doc__


def test_docstring_points_at_the_source_repository():
    assert "https://github.com/py-econometrics/pyfixest" in pf.__doc__


def test_docstring_points_at_the_installable_skill():
    assert "skills/pyfixest/SKILL.md" in pf.__doc__
    assert "https://pyfixest.org/skills.html" in pf.__doc__
