from pathlib import Path

import pytest

from scripts.bundle_agent_docs import AgentDocsBundleError, bundle

LLMS_TXT = """\
# PyFixest

> Fast high-dimensional fixed effects regression in Python.

## Pages

- [Guide](guide.llms.md)
- [Tutorials](tutorials/index.llms.md)
- [Tutorial](tutorials/tutorial.llms.md)
- [estimation.feols](reference/estimation.feols.llms.md)
- [PyFixest](index.llms.md)
"""

GUIDE_SOURCE = """\
---
title: "Guide"
description: "How to fit  models with PyFixest's 'feols' API."
categories: [guide]
---

# Guide
"""

TUTORIALS_INDEX_SOURCE = """\
---
title: "Tutorials"
---

# Tutorials
"""

TUTORIAL_SOURCE = "# Tutorial\n"

GUIDE = """\
# Guide

![Benchmark chart](figures/bench.png)

![](figures/logo.png)

![](figures/plot.png){#fig-plot fig-alt="Residuals against fitted values" width=80%}

<img src="figures/logo.svg" alt="Project logo">

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit)

<a href="https://pypi.org/project/pyfixest/"><img src="https://img.shields.io/pypi/v/pyfixest.svg" alt="PyPI"></a>

```python
print("![alt](inline.png)")
```

Done.
"""

FENCED_BLOCK = '```python\nprint("![alt](inline.png)")\n```'

CARGO_TOML = """\
[package]
name = "pyfixest_core"
version = "1.2.3"
edition = "2021"

[dependencies.pyo3]
version = "9.9.9"
features = ["extension-module"]
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def site(tmp_path: Path) -> Path:
    site = tmp_path / "_site"
    _write(site / "llms.txt", LLMS_TXT)
    _write(site / "guide.llms.md", GUIDE)
    _write(site / "tutorials" / "index.llms.md", "# Tutorials\n")
    _write(site / "tutorials" / "tutorial.llms.md", "# Tutorial\n")
    _write(site / "reference" / "estimation.feols.llms.md", "# feols\n")
    _write(site / "index.llms.md", "# PyFixest\n\nGo to [docs](guide.llms.md).\n")
    return site


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    """Build a source tree where only the guide has a front-matter description."""
    docs = tmp_path / "docs"
    _write(docs / "guide.qmd", GUIDE_SOURCE)
    _write(docs / "tutorials" / "index.qmd", TUTORIALS_INDEX_SOURCE)
    _write(docs / "tutorials" / "tutorial.md", TUTORIAL_SOURCE)
    return docs


@pytest.fixture
def version_file(tmp_path: Path) -> Path:
    path = tmp_path / "Cargo.toml"
    _write(path, CARGO_TOML)
    return path


def _bundle(site: Path, tmp_path: Path, version_file: Path, docs_dir: Path) -> Path:
    output = tmp_path / "bundled"
    bundle(site=site, output=output, version_file=version_file, docs_dir=docs_dir)
    return output


def test_bundle_copies_pages_without_the_redirect_stub(
    site: Path, tmp_path: Path, version_file: Path, docs_dir: Path
) -> None:
    output = _bundle(site, tmp_path, version_file, docs_dir)

    assert sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    ) == [
        "guide.llms.md",
        "llms.txt",
        "reference/estimation.feols.llms.md",
        "tutorials/index.llms.md",
        "tutorials/tutorial.llms.md",
    ]


def test_index_drops_the_stub_entry_and_stamps_the_version(
    site: Path, tmp_path: Path, version_file: Path, docs_dir: Path
) -> None:
    index = (_bundle(site, tmp_path, version_file, docs_dir) / "llms.txt").read_text(
        encoding="utf-8"
    )

    assert index == (
        "# PyFixest\n"
        "\n"
        "> Fast high-dimensional fixed effects regression in Python.\n"
        "\n"
        "Version: 1.2.3\n"
        "\n"
        "## Pages\n"
        "\n"
        "- [Guide](guide.llms.md): How to fit models with PyFixest's 'feols' API.\n"
        "- [Tutorials](tutorials/index.llms.md)\n"
        "- [Tutorial](tutorials/tutorial.llms.md)\n"
        "- [estimation.feols](reference/estimation.feols.llms.md)\n"
    )


def test_index_entries_carry_the_source_description(
    site: Path, tmp_path: Path, version_file: Path, docs_dir: Path
) -> None:
    index = (_bundle(site, tmp_path, version_file, docs_dir) / "llms.txt").read_text(
        encoding="utf-8"
    )

    # Quotes are stripped, inner quotes and repeated whitespace are not mangled.
    assert (
        "- [Guide](guide.llms.md): How to fit models with PyFixest's 'feols' API."
        in index
    )
    # Front matter without a description, and a page without front matter.
    assert "- [Tutorials](tutorials/index.llms.md)\n" in index
    assert "- [Tutorial](tutorials/tutorial.llms.md)\n" in index
    # quartodoc reference pages have no source to describe them.
    assert "- [estimation.feols](reference/estimation.feols.llms.md)\n" in index


def test_images_become_placeholders_and_badges_disappear(
    site: Path, tmp_path: Path, version_file: Path, docs_dir: Path
) -> None:
    guide = (
        _bundle(site, tmp_path, version_file, docs_dir) / "guide.llms.md"
    ).read_text(encoding="utf-8")

    assert "[Figure: Benchmark chart]" in guide
    assert "[Figure]" in guide
    assert "[Figure: Residuals against fitted values]" in guide
    assert "[Figure: Project logo]" in guide
    assert "fig-alt" not in guide
    assert "shields.io" not in guide
    assert "<img" not in guide
    assert "<a href" not in guide


def test_code_blocks_stay_literal(
    site: Path, tmp_path: Path, version_file: Path, docs_dir: Path
) -> None:
    guide = (
        _bundle(site, tmp_path, version_file, docs_dir) / "guide.llms.md"
    ).read_text(encoding="utf-8")

    assert FENCED_BLOCK in guide


def test_bundling_twice_is_idempotent(
    site: Path, tmp_path: Path, version_file: Path, docs_dir: Path
) -> None:
    output = _bundle(site, tmp_path, version_file, docs_dir)
    first = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }

    assert (
        bundle(site=site, output=output, version_file=version_file, docs_dir=docs_dir)
        == 4
    )
    assert {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    } == first


def test_manifest_without_a_package_version_raises(
    site: Path, tmp_path: Path, docs_dir: Path
) -> None:
    manifest = tmp_path / "no-version.toml"
    _write(
        manifest,
        '[package]\nname = "pyfixest_core"\n\n[dependencies.pyo3]\nversion = "9.9.9"\n',
    )

    with pytest.raises(AgentDocsBundleError, match=r"No \[package\] version"):
        bundle(
            site=site,
            output=tmp_path / "bundled",
            version_file=manifest,
            docs_dir=docs_dir,
        )


def test_missing_llms_index_raises(
    tmp_path: Path, version_file: Path, docs_dir: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(AgentDocsBundleError, match="Missing Quarto llms index"):
        bundle(
            site=empty,
            output=tmp_path / "bundled",
            version_file=version_file,
            docs_dir=docs_dir,
        )


def test_site_without_pages_raises(
    tmp_path: Path, version_file: Path, docs_dir: Path
) -> None:
    stub_only = tmp_path / "stub-only"
    _write(stub_only / "llms.txt", LLMS_TXT)
    _write(stub_only / "index.llms.md", "# PyFixest\n")

    with pytest.raises(AgentDocsBundleError, match=r"No Quarto \.llms\.md pages"):
        bundle(
            site=stub_only,
            output=tmp_path / "bundled",
            version_file=version_file,
            docs_dir=docs_dir,
        )
