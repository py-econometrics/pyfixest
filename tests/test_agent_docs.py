from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_agent_docs import AgentDocsError, check_agent_docs


def _write_site(path: Path, *, link: str = "guide.llms.md") -> None:
    path.mkdir()
    (path / "llms.txt").write_text(f"- [Guide]({link})\n", encoding="utf-8")
    (path / "guide.llms.md").write_text("The needle is here.\n", encoding="utf-8")


def test_agent_docs_rejects_index_without_pages(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site, link="guide.html")
    (site / "guide.html").write_text("<p>Guide</p>\n", encoding="utf-8")

    with pytest.raises(AgentDocsError, match=r"does not index any \.llms\.md"):
        check_agent_docs(site=site)


def test_agent_docs_rejects_missing_indexed_page(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)
    (site / "guide.llms.md").unlink()

    with pytest.raises(AgentDocsError, match="indexes a missing page"):
        check_agent_docs(site=site)


def test_agent_docs_rejects_unindexed_rendered_page(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)
    (site / "orphan.llms.md").write_text("# Orphan\n", encoding="utf-8")

    with pytest.raises(AgentDocsError, match=r"missing from llms\.txt"):
        check_agent_docs(site=site)


def test_agent_docs_rejects_broken_internal_link(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)
    (site / "guide.llms.md").write_text(
        "The needle is here. [Missing](missing.llms.md)\n", encoding="utf-8"
    )

    with pytest.raises(AgentDocsError, match="Broken internal link"):
        check_agent_docs(site=site)


@pytest.mark.parametrize("location", ["index", "page"])
def test_agent_docs_rejects_path_outside_site(tmp_path: Path, location: str) -> None:
    site = tmp_path / "site"
    _write_site(site)
    page = "llms.txt" if location == "index" else "guide.llms.md"
    (site / page).write_text("[Outside](../outside.llms.md)\n", encoding="utf-8")

    with pytest.raises(AgentDocsError, match="escapes the rendered site"):
        check_agent_docs(site=site)


def test_agent_docs_accepts_external_targets(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)
    (site / "guide.llms.md").write_text(
        "[Web](https://pyfixest.org/missing.html) [Mail](mailto:nobody@example.com)"
        " [Author](nobody@example.com)\n",
        encoding="utf-8",
    )

    check_agent_docs(site=site)


def test_agent_docs_accepts_root_relative_link(tmp_path: Path) -> None:
    site = tmp_path / "site"
    _write_site(site)
    (site / "llms.txt").write_text(
        "- [Guide](guide.llms.md)\n- [How to](how-to/setup.llms.md)\n", encoding="utf-8"
    )
    (site / "how-to").mkdir()
    (site / "how-to" / "setup.llms.md").write_text(
        "See the [guide](/guide.llms.md) and the [site](/index.html).\n",
        encoding="utf-8",
    )
    (site / "index.html").write_text("<p>Home</p>\n", encoding="utf-8")

    check_agent_docs(site=site)
