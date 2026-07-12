#!/usr/bin/env python3
"""Build PyFixest's deterministic, agent-readable documentation corpus.

The JSON document in ``docs/_agent_docs.yml`` is valid YAML 1.2 and is the only
page inventory. This script converts the inventoried Quarto/Markdown sources to
plain GitHub-flavoured Markdown, writes version-matched package assets, and writes
web ``.html.md`` alternates and ``llms-full.txt`` when a rendered site is present.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import posixpath
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
INVENTORY_PATH = DOCS_DIR / "_agent_docs.yml"
PACKAGE_DOCS_DIR = REPO_ROOT / "pyfixest" / "docs"
DEFAULT_SITE_DIR = DOCS_DIR / "_site"
GENERATOR_LABEL = "docs/_scripts/generate_llms_md.py"

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
REFERENCE_IMAGE_LINK_RE = re.compile(r"\[!\[([^\]]*)\]\[[^\]]+\]\]\[[^\]]+\]")
REFERENCE_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\[[^\]]+\]")
HTML_ALTERNATE_RE = re.compile(
    r"<link(?=[^>]*\brel=[\"']alternate[\"'])"
    r"(?=[^>]*\btype=[\"']text/markdown[\"'])[^>]*>",
    re.IGNORECASE,
)
RAW_HTML_FENCE_RE = re.compile(
    r"^\s*```\{=html\}\s*$.*?^\s*```\s*$", re.MULTILINE | re.DOTALL
)
SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
HTML_IMAGE_RE = re.compile(r"<(?:img|source)\b[^>]*>", re.IGNORECASE)
HTML_MEDIA_RE = re.compile(r"<(video|audio)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
HEADING_ATTRIBUTES_RE = re.compile(r"^(#{1,6}\s+.*?)\s+\{[^{}]*\}\s*$")
QUARTO_DIV_RE = re.compile(r"^\s*:::\s*(?:\{.*\})?\s*$")
QUARTO_CELL_OPTION_RE = re.compile(r"^\s*#\|.*$")
QUARTO_FENCE_RE = re.compile(
    r"^(?P<indent>\s*)```\{\.?(?P<lang>[A-Za-z0-9_+-]+)[^}]*\}\s*$"
)
QUARTO_CLASS_FENCE_RE = re.compile(
    r"^(?P<indent>\s*)```\s*\{\.(?P<lang>[A-Za-z0-9_+-]+)(?:\s+[^}]*)?\}\s*$"
)
GFM_FENCE_RE = re.compile(r"^(?P<indent>\s*)```\s+(?P<lang>[A-Za-z0-9_+-]+)\s*$")
BLOCK_ATTRIBUTE_RE = re.compile(r"^\s*\{#[^{}]+\}\s*$")
QUARTO_LEAK_RE = re.compile(
    r"(^|\n)\s*:::\s*(\{|$)|```\{(?:\.|=)?[A-Za-z]|(^|\n)\s*#\|",
    re.MULTILINE,
)
BINARY_SUFFIXES = {
    ".avif",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".svg",
    ".webp",
    ".ipynb",
}


class AgentDocsError(RuntimeError):
    """Report invalid inventory entries or generated documentation."""


@dataclass(frozen=True)
class Page:
    """One explicitly inventoried agent-documentation page."""

    source: str
    route: str
    title: str
    description: str
    section: str
    optional: bool

    @property
    def source_path(self) -> Path:
        """Return the absolute source path."""
        return REPO_ROOT / self.source

    @property
    def package_path(self) -> Path:
        """Return the absolute generated package path."""
        return PACKAGE_DOCS_DIR / "pages" / f"{self.route}.md"


@dataclass(frozen=True)
class Inventory:
    """Validated corpus metadata and its ordered pages."""

    site_url: str
    package_name: str
    summary: str
    pages: tuple[Page, ...]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_inventory_document(path: Path = INVENTORY_PATH) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentDocsError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise AgentDocsError(f"{path} must contain a mapping")
    return document


def _frontmatter(text: str) -> dict[str, str]:
    match = FRONTMATTER_RE.match(text)
    if match is None:
        return {}
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if not separator or key.strip() not in {"title", "description"}:
            continue
        value = value.strip()
        if value.startswith(('"', "'")) and value.endswith(value[0]):
            value = value[1:-1]
        metadata[key.strip()] = value
    return metadata


def load_inventory(path: Path = INVENTORY_PATH) -> Inventory:
    """Load and validate the explicit page inventory."""
    document = _read_inventory_document(path)
    missing_root = {
        "site_url",
        "package_name",
        "summary",
        "sections",
        "optional",
    } - document.keys()
    if missing_root:
        raise AgentDocsError(
            f"{path} is missing keys: {', '.join(sorted(missing_root))}"
        )

    rows: list[tuple[dict[str, Any], str, bool]] = []
    for section in document["sections"]:
        if not isinstance(section, dict) or not section.get("title"):
            raise AgentDocsError("Every primary section needs a title")
        for entry in section.get("pages", []):
            rows.append((entry, str(section["title"]), False))
    for entry in document["optional"]:
        rows.append((entry, "Optional", True))

    pages: list[Page] = []
    seen_sources: set[str] = set()
    seen_routes: set[str] = set()
    for entry, section, optional in rows:
        if not isinstance(entry, dict):
            raise AgentDocsError(f"Invalid page entry in {section}: {entry!r}")
        source = str(entry.get("source", "")).strip()
        route = str(entry.get("route", "")).strip().strip("/")
        if not source or not route:
            raise AgentDocsError(f"Every page in {section} needs source and route")
        if route.endswith((".html", ".md")) or ".." in PurePosixPath(route).parts:
            raise AgentDocsError(f"Invalid route {route!r}")
        if source in seen_sources:
            raise AgentDocsError(f"Duplicate source {source!r}")
        if route.casefold() in seen_routes:
            raise AgentDocsError(f"Duplicate route {route!r}")
        seen_sources.add(source)
        seen_routes.add(route.casefold())

        source_path = REPO_ROOT / source
        if not source_path.is_file():
            hint = (
                " Run `pixi run docs-build` first." if "/reference/" in source else ""
            )
            raise AgentDocsError(f"Missing inventoried source {source}.{hint}")
        raw = source_path.read_text(encoding="utf-8")
        metadata = _frontmatter(raw)
        title = str(entry.get("title") or metadata.get("title") or "").strip()
        description = str(
            entry.get("description") or metadata.get("description") or ""
        ).strip()
        if not title or not description:
            missing = "title" if not title else "description"
            raise AgentDocsError(f"{source} has no {missing} metadata")
        pages.append(Page(source, route, title, description, section, optional))

    if not pages:
        raise AgentDocsError("The agent-documentation inventory is empty")
    return Inventory(
        site_url=str(document["site_url"]).rstrip("/"),
        package_name=str(document["package_name"]),
        summary=str(document["summary"]),
        pages=tuple(pages),
    )


def _strip_frontmatter(text: str) -> str:
    return FRONTMATTER_RE.sub("", text, count=1)


def _source_key(source: str) -> str:
    return posixpath.normpath(source.replace(os.sep, "/"))


def _source_aliases(page: Page) -> set[str]:
    aliases = {_source_key(page.source)}
    if page.source == "ARCHITECTURE.md":
        aliases.add("docs/architecture.qmd")
    return aliases


def _resolve_internal_page(
    target: str,
    current: Page,
    inventory: Inventory,
) -> tuple[Page, str] | None:
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        if f"{parsed.scheme}://{parsed.netloc}" != inventory.site_url:
            return None
        route = unquote(parsed.path).lstrip("/")
        if route.endswith(".html.md"):
            route = route[: -len(".html.md")]
        elif route.endswith(".html"):
            route = route[: -len(".html")]
        route_map = {page.route: page for page in inventory.pages}
        page = route_map.get(route)
        return (page, parsed.fragment) if page else None

    raw_path = unquote(parsed.path)
    if not raw_path or raw_path.startswith("#"):
        return None
    if raw_path.startswith("/"):
        source = _source_key(f"docs/{raw_path.lstrip('/')}")
    else:
        source = _source_key(
            posixpath.join(posixpath.dirname(current.source), raw_path)
        )
    source_map = {
        alias: page for page in inventory.pages for alias in _source_aliases(page)
    }
    source_candidates = [source]
    suffix = PurePosixPath(source).suffix
    if suffix in {".md", ".qmd"}:
        alternate_suffix = ".qmd" if suffix == ".md" else ".md"
        source_candidates.append(f"{source[: -len(suffix)]}{alternate_suffix}")
    if not source.startswith("docs/"):
        source_candidates.extend(
            [f"docs/{candidate}" for candidate in source_candidates]
        )
    page = next(
        (
            source_map[candidate]
            for candidate in source_candidates
            if candidate in source_map
        ),
        None,
    )
    if page is None and raw_path.endswith(".html"):
        route = raw_path[: -len(".html")].lstrip("/")
        page = next((item for item in inventory.pages if item.route == route), None)
    return (page, parsed.fragment) if page else None


def _rewrite_links(
    text: str,
    page: Page,
    inventory: Inventory,
    destination: str,
) -> str:
    def image_replacement(match: re.Match[str]) -> str:
        alt = match.group(1).strip()
        return f"*Image omitted: {alt}*" if alt else ""

    def link_replacement(match: re.Match[str]) -> str:
        label, target = match.groups()
        target = target.strip().strip("<>")
        if "@" in target and ":" not in target and "/" not in target:
            return f"[{label}](mailto:{target})"
        parsed = urlsplit(target)
        if PurePosixPath(parsed.path).suffix.casefold() in BINARY_SUFFIXES:
            return f"{label} *(binary asset omitted)*"
        resolved = _resolve_internal_page(target, page, inventory)
        if resolved is None:
            return match.group(0)
        target_page, fragment = resolved
        suffix = f"#{fragment}" if fragment else ""
        if destination == "web":
            href = f"{inventory.site_url}/{target_page.route}.html.md{suffix}"
        else:
            current_dir = posixpath.dirname(page.route) or "."
            href = posixpath.relpath(f"{target_page.route}.md", current_dir)
            href = f"{href}{suffix}"
        return f"[{label}]({href})"

    output: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            output.append(line)
            continue
        if not in_fence:
            line = REFERENCE_IMAGE_LINK_RE.sub(
                lambda match: f"*Image omitted: {match.group(1).strip()}*", line
            )
            line = REFERENCE_IMAGE_RE.sub(
                lambda match: f"*Image omitted: {match.group(1).strip()}*", line
            )
            line = MARKDOWN_IMAGE_RE.sub(image_replacement, line)
            line = MARKDOWN_LINK_RE.sub(link_replacement, line)
        output.append(line)
    return "\n".join(output)


def source_to_gfm(
    page: Page,
    inventory: Inventory,
    destination: str,
) -> str:
    """Convert an inventoried Quarto/Markdown source to stable plain GFM."""
    text = page.source_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    text = _strip_frontmatter(text)
    text = RAW_HTML_FENCE_RE.sub("", text)
    text = SCRIPT_STYLE_RE.sub("", text)
    text = HTML_IMAGE_RE.sub("*Image omitted.*", text)
    text = HTML_MEDIA_RE.sub("*Binary media omitted.*", text)

    cleaned: list[str] = []
    for line in text.splitlines():
        if QUARTO_DIV_RE.match(line) or BLOCK_ATTRIBUTE_RE.match(line):
            continue
        if QUARTO_CELL_OPTION_RE.match(line):
            continue
        fence = (
            QUARTO_FENCE_RE.match(line)
            or QUARTO_CLASS_FENCE_RE.match(line)
            or GFM_FENCE_RE.match(line)
        )
        if fence:
            cleaned.append(f"{fence.group('indent')}```{fence.group('lang')}")
            continue
        heading = HEADING_ATTRIBUTES_RE.match(line)
        if heading:
            line = heading.group(1)
        cleaned.append(line.rstrip())

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = _rewrite_links(text, page, inventory, destination)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    if QUARTO_LEAK_RE.search(text):
        raise AgentDocsError(f"Quarto syntax leaked from {page.source}")
    header = f"<!-- Generated from {page.source}; do not edit. -->\n\n"
    return header + text


def _llms_index(inventory: Inventory, destination: str) -> str:
    lines = [f"# {inventory.package_name}", "", f"> {inventory.summary}"]
    sections: list[str] = []
    for page in inventory.pages:
        if page.section not in sections:
            sections.append(page.section)
    for section in sections:
        lines.extend(["", f"## {section}", ""])
        for page in (item for item in inventory.pages if item.section == section):
            if destination == "web":
                href = f"{inventory.site_url}/{page.route}.html.md"
            else:
                href = f"pages/{page.route}.md"
            lines.append(f"- [{page.title}]({href}): {page.description}")
    return "\n".join(lines).rstrip() + "\n"


def _package_index(inventory: Inventory) -> str:
    lines = [
        "<!-- Generated from docs/_agent_docs.yml; do not edit. -->",
        "",
        f"# {inventory.package_name} documentation",
        "",
        inventory.summary,
        "",
        "These pages match the installed PyFixest version and work offline. Start with",
        "`llms.txt` for a compact routing index.",
    ]
    sections: list[str] = []
    for page in inventory.pages:
        if page.section not in sections:
            sections.append(page.section)
    for section in sections:
        lines.extend(["", f"## {section}", ""])
        for page in (item for item in inventory.pages if item.section == section):
            lines.append(f"- [{page.title}](pages/{page.route}.md): {page.description}")
    return "\n".join(lines).rstrip() + "\n"


def _manifest(
    inventory: Inventory,
    package_pages: dict[Path, str],
) -> str:
    rows: list[dict[str, Any]] = []
    for page in inventory.pages:
        content = package_pages[page.package_path]
        rows.append(
            {
                "content_sha256": _sha256(content),
                "description": page.description,
                "optional": page.optional,
                "route": page.route,
                "section": page.section,
                "source": page.source,
                "source_sha256": _sha256(page.source_path.read_text(encoding="utf-8")),
                "title": page.title,
            }
        )
    corpus_hash = hashlib.sha256()
    for row in rows:
        corpus_hash.update(row["route"].encode())
        corpus_hash.update(row["content_sha256"].encode())
    document = {
        "corpus_sha256": corpus_hash.hexdigest(),
        "generator": GENERATOR_LABEL,
        "pages": rows,
        "schema_version": 1,
        "site_url": inventory.site_url,
    }
    return json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def expected_package_outputs(inventory: Inventory) -> dict[Path, str]:
    """Build every expected tracked package output in memory."""
    outputs = {
        page.package_path: source_to_gfm(page, inventory, "package")
        for page in inventory.pages
    }
    outputs[PACKAGE_DOCS_DIR / "index.md"] = _package_index(inventory)
    outputs[PACKAGE_DOCS_DIR / "llms.txt"] = _llms_index(inventory, "package")
    outputs[PACKAGE_DOCS_DIR / "manifest.json"] = _manifest(inventory, outputs)
    return outputs


def _compare_or_write(outputs: dict[Path, str], check: bool) -> list[str]:
    problems: list[str] = []
    for path, expected in outputs.items():
        actual = path.read_text(encoding="utf-8") if path.is_file() else None
        if actual == expected:
            continue
        if check:
            try:
                display_path = path.relative_to(REPO_ROOT)
            except ValueError:
                display_path = path
            problems.append(f"stale or missing: {display_path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")
    return problems


def _remove_or_report_stale_package_files(
    expected: set[Path],
    check: bool,
) -> list[str]:
    if not PACKAGE_DOCS_DIR.exists():
        return []
    candidates = {
        path
        for path in PACKAGE_DOCS_DIR.rglob("*")
        if path.is_file() and path.suffix in {".md", ".txt", ".json"}
    }
    stale = sorted(candidates - expected)
    if check:
        return [
            f"unexpected generated file: {path.relative_to(REPO_ROOT)}"
            for path in stale
        ]
    for path in stale:
        path.unlink()
    return []


def _inject_alternate(html_path: Path, markdown_name: str, check: bool) -> list[str]:
    text = html_path.read_text(encoding="utf-8")
    link = (
        '<link rel="alternate" type="text/markdown" '
        f'href="{html.escape(markdown_name, quote=True)}">'
    )
    match = HTML_ALTERNATE_RE.search(text)
    if match:
        expected = text[: match.start()] + link + text[match.end() :]
    elif "</head>" in text:
        expected = text.replace("</head>", f"{link}\n</head>", 1)
    else:
        return [f"HTML page has no </head>: {html_path.relative_to(REPO_ROOT)}"]
    if expected == text:
        return []
    if check:
        return [f"stale alternate link: {html_path.relative_to(REPO_ROOT)}"]
    html_path.write_text(expected, encoding="utf-8")
    return []


def expected_site_outputs(
    inventory: Inventory,
    site_dir: Path,
) -> dict[Path, str]:
    """Build expected web alternates, llms indexes, and site-only full corpus."""
    outputs = {
        site_dir / f"{page.route}.html.md": source_to_gfm(page, inventory, "web")
        for page in inventory.pages
    }
    web_index = _llms_index(inventory, "web")
    outputs[DOCS_DIR / "llms.txt"] = web_index
    outputs[site_dir / "llms.txt"] = web_index
    full_parts: list[str] = []
    for page in inventory.pages:
        body = outputs[site_dir / f"{page.route}.html.md"].strip()
        full_parts.append(f"<!-- {page.route}.html -->\n\n{body}")
    outputs[site_dir / "llms-full.txt"] = "\n\n---\n\n".join(full_parts) + "\n"
    return outputs


def _validate_package_links(outputs: dict[Path, str]) -> list[str]:
    expected = set(outputs)
    problems: list[str] = []
    for path, content in outputs.items():
        if path.suffix not in {".md", ".txt"}:
            continue
        for _, target in MARKDOWN_LINK_RE.findall(content):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            resolved = (path.parent / unquote(parsed.path)).resolve()
            if resolved not in expected:
                problems.append(
                    f"broken local link in {path.relative_to(REPO_ROOT)}: {target}"
                )
    return problems


def generate(
    check: bool,
    site_dir: Path | None,
) -> list[str]:
    """Generate or check all package and optional site outputs."""
    inventory = load_inventory()
    package_outputs = expected_package_outputs(inventory)
    problems = _validate_package_links(package_outputs)
    problems.extend(_compare_or_write(package_outputs, check))
    problems.extend(_remove_or_report_stale_package_files(set(package_outputs), check))

    web_index = _llms_index(inventory, "web")
    problems.extend(_compare_or_write({DOCS_DIR / "llms.txt": web_index}, check))
    if site_dir is not None and site_dir.exists():
        site_outputs = expected_site_outputs(inventory, site_dir)
        problems.extend(_compare_or_write(site_outputs, check))
        expected_alternates = {
            path for path in site_outputs if path.name.endswith(".html.md")
        }
        existing_alternates = set(site_dir.rglob("*.html.md"))
        stale_alternates = sorted(existing_alternates - expected_alternates)
        if check:
            problems.extend(
                f"unexpected site alternate: {path.relative_to(site_dir)}"
                for path in stale_alternates
            )
        else:
            for path in stale_alternates:
                path.unlink()
        for page in inventory.pages:
            html_path = site_dir / f"{page.route}.html"
            if not html_path.is_file():
                problems.append(
                    f"missing rendered HTML page: {html_path.relative_to(site_dir)}"
                )
                continue
            problems.extend(_inject_alternate(html_path, f"{html_path.name}.md", check))
    return problems


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of rewriting stale generated assets.",
    )
    parser.add_argument(
        "--no-site",
        action="store_true",
        help="Generate/check tracked package and llms.txt assets only.",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=DEFAULT_SITE_DIR,
        help="Rendered Quarto site root (default: docs/_site).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line generator."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    site_dir = None if args.no_site else args.site_dir.resolve()
    try:
        problems = generate(args.check, site_dir)
    except AgentDocsError as exc:
        print(f"agent docs: {exc}", file=sys.stderr)
        return 1
    if problems:
        print("agent docs validation failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    verb = "checked" if args.check else "generated"
    print(f"agent docs: {verb} package corpus and llms.txt")
    if site_dir is not None and site_dir.exists():
        print(f"agent docs: {verb} Markdown alternates in {site_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
