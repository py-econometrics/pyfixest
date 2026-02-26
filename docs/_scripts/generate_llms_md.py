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
ALT_LINK_RE = re.compile(
    r"<link(?=[^>]*\\brel=[\"']alternate[\"'])"
    r"(?=[^>]*\\btype=[\"']text/markdown[\"'])[^>]*>",
    re.IGNORECASE,
)


def strip_yaml_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (``---`` … ``---``) from the start of a file."""
    return re.sub(r"\A---\n.*?\n---\n*", "", text, count=1, flags=re.DOTALL)


MAX_OUTPUT_CHARS = 4000
MAX_CELL_OUTPUT_CHARS = 8000


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def _format_output(output: dict) -> list[str]:
    parts: list[str] = []
    output_type = output.get("output_type")
    if output_type == "stream":
        text = output.get("text", "")
        if text:
            parts.append(f"```text\n{_truncate(text, MAX_OUTPUT_CHARS)}\n```")
    elif output_type in ("execute_result", "display_data"):
        data = output.get("data", {})
        if "text/markdown" in data:
            parts.append(_truncate("".join(data["text/markdown"]), MAX_OUTPUT_CHARS))
        elif "text/plain" in data:
            parts.append(
                f"```text\n{_truncate(''.join(data['text/plain']), MAX_OUTPUT_CHARS)}\n```"
            )
        else:
            for mime in data:
                if mime.startswith("image/"):
                    parts.append(f"[{mime} omitted]")
                    break
    elif output_type == "error":
        tb = "\n".join(output.get("traceback", []))
        if tb:
            parts.append(f"```text\n{_truncate(tb, MAX_OUTPUT_CHARS)}\n```")
    return parts


def notebook_to_markdown(nb_path: Path) -> str:
    """Extract markdown, code, and outputs from a Jupyter notebook."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    parts: list[str] = []
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        cell_type = cell.get("cell_type")
        if cell_type == "markdown":
            parts.append(source)
        elif cell_type == "code":
            parts.append(f"```python\n{source}\n```")
            output_parts: list[str] = []
            output_chars = 0
            for output in cell.get("outputs", []):
                for out_part in _format_output(output):
                    output_chars += len(out_part)
                    if output_chars > MAX_CELL_OUTPUT_CHARS:
                        output_parts.append("```text\n...[outputs truncated]...\n```")
                        break
                    output_parts.append(out_part)
                if output_chars > MAX_CELL_OUTPUT_CHARS:
                    break
            if output_parts:
                parts.extend(output_parts)
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
    """Generate LLM-friendly markdown alongside built docs."""
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

        # Inject a deterministic, relative markdown link into the HTML head.
        try:
            html_text = html_file.read_text(encoding="utf-8")
        except OSError:
            continue
        if not ALT_LINK_RE.search(html_text):
            link_tag = (
                f'<link rel="alternate" type="text/markdown" '
                f'href="{html_file.name}.md">'
            )
            if "</head>" in html_text:
                html_text = html_text.replace("</head>", f"{link_tag}\n</head>", 1)
                html_file.write_text(html_text, encoding="utf-8")

    print(f"generate_llms_md: wrote {count} .html.md files.")

    # Generate llms-full.txt by concatenating all .html.md files
    md_files = sorted(SITE_DIR.rglob("*.html.md"))
    if md_files:
        parts: list[str] = []
        for md_file in md_files:
            page_path = md_file.relative_to(SITE_DIR)
            # Section header uses the HTML page name (strip trailing .md)
            html_name = str(page_path)[: -len(".md")]
            content = md_file.read_text(encoding="utf-8").strip()
            parts.append(f"# {html_name}\n\n{content}")
        full_txt = "\n\n---\n\n".join(parts) + "\n"
        (SITE_DIR / "llms-full.txt").write_text(full_txt, encoding="utf-8")
        print(
            f"generate_llms_md: wrote llms-full.txt ({len(md_files)} pages, "
            f"{len(full_txt)} chars)."
        )

    # Copy llms.txt into _site/
    llms_src = DOCS_DIR / "llms.txt"
    if llms_src.exists():
        shutil.copy2(llms_src, SITE_DIR / "llms.txt")
        print("generate_llms_md: copied llms.txt to _site/.")


if __name__ == "__main__":
    main()
