#!/usr/bin/env python3
"""Quarto post-render script: generate .html.md files for llms.txt compliance.

For every .html file in _site/, this script finds the corresponding source file
(.qmd, .md, or .ipynb) under docs/, strips YAML frontmatter, and writes a clean
markdown version as <name>.html.md alongside the HTML file.

It also copies docs/llms.txt into _site/.
"""

import json
import re
import shutil
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent  # docs/
SITE_DIR = DOCS_DIR / "_site"


def strip_yaml_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (``---`` … ``---``) from the start of a file."""
    return re.sub(r"\A---\n.*?\n---\n*", "", text, count=1, flags=re.DOTALL)


def notebook_to_markdown(nb_path: Path) -> str:
    """Extract markdown and code cells from a Jupyter notebook."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    parts: list[str] = []
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            parts.append(source)
        elif cell["cell_type"] == "code":
            parts.append(f"```python\n{source}\n```")
    return "\n\n".join(parts)


def find_source(html_rel: Path) -> Path | None:
    """Map a site-relative HTML path back to its docs source file.

    For example ``quickstart.html`` → ``docs/quickstart.qmd``.

    Uses string manipulation instead of Path.with_suffix() because dotted
    filenames like ``did.estimation.did2s.html`` confuse Path's suffix handling.
    """
    # Strip the .html extension via string ops to handle dotted names
    rel_str = str(html_rel)
    if rel_str.endswith(".html"):
        stem_str = rel_str[: -len(".html")]
    else:
        return None

    for ext in (".qmd", ".md", ".ipynb"):
        candidate = DOCS_DIR / (stem_str + ext)
        if candidate.exists():
            return candidate
    # Some pages (e.g. reference/index.html) may have an index source
    for ext in (".qmd", ".md"):
        candidate = DOCS_DIR / stem_str / f"index{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    if not SITE_DIR.exists():
        print("generate_llms_md: _site/ not found, skipping.")
        return

    count = 0
    for html_file in sorted(SITE_DIR.rglob("*.html")):
        html_rel = html_file.relative_to(SITE_DIR)

        source = find_source(html_rel)
        if source is None:
            continue

        if source.suffix == ".ipynb":
            md_content = notebook_to_markdown(source)
        else:
            raw = source.read_text(encoding="utf-8")
            md_content = strip_yaml_frontmatter(raw)

        out_path = html_file.parent / f"{html_file.name}.md"
        out_path.write_text(md_content, encoding="utf-8")
        count += 1

    print(f"generate_llms_md: wrote {count} .html.md files.")

    # Copy llms.txt into _site/
    llms_src = DOCS_DIR / "llms.txt"
    if llms_src.exists():
        shutil.copy2(llms_src, SITE_DIR / "llms.txt")
        print("generate_llms_md: copied llms.txt to _site/.")


if __name__ == "__main__":
    main()
