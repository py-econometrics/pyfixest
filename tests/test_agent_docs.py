import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "docs" / "_scripts" / "generate_llms_md.py"
PACKAGE_DOCS = REPO_ROOT / "pyfixest" / "docs"
SKILL_DIR = REPO_ROOT / "skills" / "pyfixest"


@pytest.fixture(scope="module")
def agent_docs_module():
    spec = importlib.util.spec_from_file_location("generate_llms_md", GENERATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_inventory_has_unique_metadata_and_routes():
    inventory = json.loads(
        (REPO_ROOT / "docs" / "_agent_docs.yml").read_text(encoding="utf-8")
    )
    entries = [
        entry for section in inventory["sections"] for entry in section["pages"]
    ] + inventory["optional"]
    manifest = json.loads((PACKAGE_DOCS / "manifest.json").read_text(encoding="utf-8"))
    pages = manifest["pages"]
    assert len(entries) >= 60
    assert len({entry["source"] for entry in entries}) == len(entries)
    assert len({entry["route"].casefold() for entry in entries}) == len(entries)
    assert all(page["title"] and page["description"] for page in pages)
    assert any(entry["route"] == "troubleshooting" for entry in entries)


def test_packaged_corpus_matches_manifest():
    manifest = json.loads((PACKAGE_DOCS / "manifest.json").read_text(encoding="utf-8"))
    pages = manifest["pages"]
    assert manifest["schema_version"] == 1
    assert pages
    assert not (PACKAGE_DOCS / "llms-full.txt").exists()

    for page in pages:
        path = PACKAGE_DOCS / "pages" / f"{page['route']}.md"
        content = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(content.encode()).hexdigest()
        assert digest == page["content_sha256"]
        assert "```{python}" not in content
        assert "```{.python" not in content
        assert "```{=html}" not in content
        assert "\n:::" not in content
        assert "![" not in content
        assert "<img" not in content.casefold()


def test_llms_txt_is_only_spec_index():
    text = (REPO_ROOT / "docs" / "llms.txt").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "# PyFixest"
    assert lines[2].startswith("> ")
    assert "## Optional" in lines
    assert all(not line or line.startswith(("#", "> ", "- [")) for line in lines)


def test_quarto_post_render_uses_agent_docs_generator():
    config = (REPO_ROOT / "docs" / "_quarto.yml").read_text(encoding="utf-8")
    assert "_scripts/generate_llms_md.py" in config
    assert "_scripts/generate_llms_md.py --quarto" not in config


def test_packaged_internal_links_resolve(agent_docs_module):
    outputs = {
        path: path.read_text(encoding="utf-8")
        for path in PACKAGE_DOCS.rglob("*")
        if path.is_file()
    }
    assert agent_docs_module._validate_package_links(outputs) == []


def test_check_mode_reports_stale_output(agent_docs_module, tmp_path):
    output = tmp_path / "generated.md"
    assert agent_docs_module._compare_or_write({output: "expected\n"}, True)
    assert not output.exists()
    assert agent_docs_module._compare_or_write({output: "expected\n"}, False) == []
    assert agent_docs_module._compare_or_write({output: "expected\n"}, True) == []


def test_alternate_link_is_deterministic(agent_docs_module, tmp_path):
    html_path = tmp_path / "guide.html"
    html_path.write_text("<html><head></head><body></body></html>\n", encoding="utf-8")
    assert (
        agent_docs_module._inject_alternate(html_path, "guide.html.md", check=False)
        == []
    )
    content = html_path.read_text(encoding="utf-8")
    assert content.count('rel="alternate"') == 1
    assert 'href="guide.html.md"' in content
    assert (
        agent_docs_module._inject_alternate(html_path, "guide.html.md", check=True)
        == []
    )


def test_inventory_rejects_duplicate_routes(agent_docs_module, tmp_path, monkeypatch):
    source = tmp_path / "page.md"
    source.write_text(
        '---\ntitle: "Page"\ndescription: "A useful page."\n---\n',
        encoding="utf-8",
    )
    inventory = {
        "site_url": "https://example.test",
        "package_name": "Example",
        "summary": "Example corpus.",
        "sections": [
            {
                "title": "Docs",
                "pages": [
                    {"source": "page.md", "route": "same"},
                    {"source": "other.md", "route": "same"},
                ],
            }
        ],
        "optional": [],
    }
    (tmp_path / "other.md").write_text(source.read_text(encoding="utf-8"))
    config = tmp_path / "inventory.yml"
    config.write_text(json.dumps(inventory), encoding="utf-8")
    monkeypatch.setattr(agent_docs_module, "REPO_ROOT", tmp_path)
    with pytest.raises(agent_docs_module.AgentDocsError, match="Duplicate route"):
        agent_docs_module.load_inventory(config)


def test_inventory_rejects_missing_metadata(agent_docs_module, tmp_path, monkeypatch):
    (tmp_path / "page.md").write_text(
        '---\ntitle: "Page"\n---\n\nNo description.\n', encoding="utf-8"
    )
    inventory = {
        "site_url": "https://example.test",
        "package_name": "Example",
        "summary": "Example corpus.",
        "sections": [
            {
                "title": "Docs",
                "pages": [{"source": "page.md", "route": "page"}],
            }
        ],
        "optional": [],
    }
    config = tmp_path / "inventory.yml"
    config.write_text(json.dumps(inventory), encoding="utf-8")
    monkeypatch.setattr(agent_docs_module, "REPO_ROOT", tmp_path)
    with pytest.raises(agent_docs_module.AgentDocsError, match="no description"):
        agent_docs_module.load_inventory(config)


def test_canonical_skill_has_seven_references_and_generated_page(agent_docs_module):
    """Keep the bundled skill small, valid, and the sole source of its web page."""
    skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    frontmatter = skill.split("---", 2)[1].strip().splitlines()
    assert {line.split(":", 1)[0] for line in frontmatter} == {
        "name",
        "description",
    }
    assert "name: pyfixest" in skill

    references = sorted((SKILL_DIR / "references").glob("*.md"))
    assert len(references) == 7
    assert all(reference.parent == SKILL_DIR / "references" for reference in references)

    metadata = (SKILL_DIR / "agents" / "openai.yaml").read_text(encoding="utf-8")
    assert 'default_prompt: "Use $pyfixest' in metadata
    assert (REPO_ROOT / "docs" / "skills.md").read_text(
        encoding="utf-8"
    ) == agent_docs_module.expected_skill_page()
