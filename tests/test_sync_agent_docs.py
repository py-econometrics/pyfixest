from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.sync_agent_docs import AgentDocsGenerationError, generate_bundle


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    site = tmp_path / "site"
    output = tmp_path / "package-docs"
    site.mkdir()
    (site / "guide.llms.md").write_text(
        "# Guide\n\n![Diagram](plot.png)\n\n"
        "[Details](nested/details.llms.md)\n\n"
        "[API](reference/api.llms.md)\n\n"
        "``` python\nprint('kept')\n```\n\n```\nkept output\n```\n",
        encoding="utf-8",
    )
    (site / "nested").mkdir()
    (site / "nested/details.llms.md").write_text(
        "# Details\n\n[Back](../guide.llms.md)\n", encoding="utf-8"
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "site_url": "https://pyfixest.org/",
                "pages": [
                    {
                        "source": "guide.qmd",
                        "title": "Guide",
                        "description": "Start here.",
                    },
                    {
                        "source": "nested/details.qmd",
                        "title": "Details",
                        "description": "More detail.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cargo_manifest = tmp_path / "Cargo.toml"
    cargo_manifest.write_text(
        '[package]\nname = "example"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    return site, inventory, output, cargo_manifest


def test_generate_bundle_rewrites_links_and_images(tmp_path: Path) -> None:
    site, inventory, output, cargo_manifest = _write_fixture(tmp_path)

    generate_bundle(
        site=site,
        inventory_path=inventory,
        output=output,
        cargo_manifest=cargo_manifest,
    )

    guide = (output / "pages/guide.md").read_text(encoding="utf-8")
    assert "Image: Diagram" in guide
    assert "plot.png" not in guide
    assert "[Details](nested/details.md)" in guide
    assert "[API](https://pyfixest.org/reference/api.llms.md)" in guide
    assert "print('kept')" in guide
    assert "kept output" in guide
    assert "PyFixest 1.2.3" in (output / "index.md").read_text(encoding="utf-8")

    generate_bundle(
        site=site,
        inventory_path=inventory,
        output=output,
        cargo_manifest=cargo_manifest,
        check=True,
    )


def test_generate_bundle_detects_drift(tmp_path: Path) -> None:
    site, inventory, output, cargo_manifest = _write_fixture(tmp_path)
    generate_bundle(
        site=site,
        inventory_path=inventory,
        output=output,
        cargo_manifest=cargo_manifest,
    )
    (output / "pages/guide.md").write_text("stale\n", encoding="utf-8")

    with pytest.raises(AgentDocsGenerationError, match="out of date"):
        generate_bundle(
            site=site,
            inventory_path=inventory,
            output=output,
            cargo_manifest=cargo_manifest,
            check=True,
        )


def test_generate_bundle_requires_rendered_pages(tmp_path: Path) -> None:
    site, inventory, output, cargo_manifest = _write_fixture(tmp_path)
    (site / "guide.llms.md").unlink()

    with pytest.raises(AgentDocsGenerationError, match="Missing rendered page"):
        generate_bundle(
            site=site,
            inventory_path=inventory,
            output=output,
            cargo_manifest=cargo_manifest,
        )
