from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_agent_docs import AgentDocsError, check_agent_docs


def _write_cases(path: Path, *, terms: list[str] | None = None) -> Path:
    cases = [
        {
            "id": "example",
            "question": "Where is the example?",
            "authoritative_sources": ["guide.llms.md"],
            "terms": terms or ["needle"],
        }
    ]
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


def _write_site(path: Path, *, link: str = "guide.llms.md") -> None:
    path.mkdir()
    (path / "llms.txt").write_text(f"- [Guide]({link})\n", encoding="utf-8")
    (path / "guide.llms.md").write_text("The needle is here.\n", encoding="utf-8")


def test_agent_docs_rejects_malformed_index(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site, link="guide.html")
    (site / "guide.html").write_text("<p>Guide</p>\n", encoding="utf-8")

    with pytest.raises(AgentDocsError, match=r"does not index any \.llms\.md"):
        check_agent_docs(site=site, cases_path=_write_cases(tmp_path / "cases.json"))


def test_agent_docs_rejects_missing_indexed_page(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)
    (site / "guide.llms.md").unlink()

    with pytest.raises(AgentDocsError, match="indexes a missing page"):
        check_agent_docs(site=site, cases_path=_write_cases(tmp_path / "cases.json"))


def test_agent_docs_rejects_broken_internal_link(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)
    (site / "guide.llms.md").write_text(
        "The needle is here. [Missing](missing.llms.md)\n", encoding="utf-8"
    )

    with pytest.raises(AgentDocsError, match="Broken internal link"):
        check_agent_docs(site=site, cases_path=_write_cases(tmp_path / "cases.json"))


def test_agent_docs_ignores_links_shown_in_fenced_code(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)
    (site / "guide.llms.md").write_text(
        "The needle is here.\n\n````markdown\n"
        "[Example](missing.llms.md)\n\n```python\npass\n```\n````\n",
        encoding="utf-8",
    )

    check_agent_docs(site=site, cases_path=_write_cases(tmp_path / "cases.json"))


def test_agent_docs_rejects_failed_retrieval_case(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)

    with pytest.raises(AgentDocsError, match="misses terms: 'absent'"):
        check_agent_docs(
            site=site,
            cases_path=_write_cases(tmp_path / "cases.json", terms=["absent"]),
        )
