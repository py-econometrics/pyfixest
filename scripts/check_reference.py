"""Check generated API navigation and documented function parameters."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import griffe
import yaml


class _HtmlPage(HTMLParser):
    def __init__(self, path: Path):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(path.read_text())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"])


def check_rendered_links(site: Path) -> list[str]:
    """Check actual HTML IDs, including IDs consumed by Quarto title promotion."""
    site = site.resolve()
    pages = sorted((site / "reference").glob("*.html"))
    pages += sorted((site / "reference-guide").glob("*.html"))
    if not pages:
        return [f"No rendered reference pages in {site}; render the site first"]
    parsed = {page: _HtmlPage(page) for page in pages}
    issues = set()
    for source, page in list(parsed.items()):
        for link in page.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            target = (
                (
                    site / parts.path.lstrip("/")
                    if parts.path.startswith("/")
                    else source.parent / parts.path
                ).resolve()
                if parts.path
                else source
            )
            if target.is_dir():
                target /= "index.html"
            if target.suffix not in {".html", ".qmd"}:
                continue
            if not target.is_file() or target.suffix == ".qmd":
                issues.add(f"{source.relative_to(site)}: missing HTML target {link}")
                continue
            if parts.fragment:
                if target not in parsed:
                    parsed[target] = _HtmlPage(target)
                if unquote(parts.fragment) not in parsed[target].ids:
                    issues.add(
                        f"{source.relative_to(site)}: missing HTML anchor {link}"
                    )
    return sorted(issues)


def registered_pages(config: dict) -> list[tuple[str, list[str]]]:
    """Read explicit page destinations, including combined object pages."""
    pages = []
    for section in config["quartodoc"]["sections"]:
        for entry in section["contents"]:
            if isinstance(entry, str):
                pages.append((entry, [entry]))
            elif entry.get("kind") == "page":
                names = [
                    item if isinstance(item, str) else item["name"]
                    for item in entry["contents"]
                ]
                pages.append((entry["path"], names))
            else:
                pages.append((entry["name"], [entry["name"]]))
    return pages


def _prose(text: str) -> str:
    return re.sub(r"(?ms)^([ \t]*)```[^\n]*\n.*?^\1```[^\n]*$", "", text)


def _anchors(text: str) -> set[str]:
    text = _prose(text)
    anchors = set(re.findall(r"\{\s*#([^\s}]+)", text))
    anchors.update(re.findall(r'<[^>]+\bid=["\']([^"\']+)', text))
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+)$", text):
        if re.search(r"\{[^}]*#[^\s}]+[^}]*\}\s*$", heading):
            continue
        heading = re.sub(r"\{[^}]*\}\s*$", "", heading)
        heading = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", heading)
        slug = re.sub(r"[^\w\s-]", "", heading.lower()).strip()
        anchors.add(re.sub(r"\s+", "-", slug))
    return anchors


def check_links(docs: Path, pages: list[Path]) -> list[str]:
    """Resolve local destinations and fragments from source Markdown."""
    issues = []
    for page in pages:
        text = _prose(page.read_text())
        for match in re.finditer(r"\]\(([^)\s]+)\)", text):
            link = match.group(1)
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            target = (
                docs / parsed.path.lstrip("/")
                if parsed.path.startswith("/")
                else page.parent / parsed.path
            )
            if not parsed.path:
                target = page
            if target.suffix == ".html":
                target = next(
                    (
                        target.with_suffix(suffix)
                        for suffix in (".qmd", ".md", ".ipynb")
                        if target.with_suffix(suffix).exists()
                    ),
                    target,
                )
            if not target.is_file():
                issues.append(f"{page.relative_to(docs)}: missing target {link}")
            elif (
                parsed.fragment
                and target.suffix in {".qmd", ".md"}
                and unquote(parsed.fragment) not in _anchors(target.read_text())
            ):
                issues.append(f"{page.relative_to(docs)}: missing anchor {link}")
    return issues


def check_parameters(package, names: list[str]) -> list[str]:
    """Compare public signature names with NumPy parameter sections."""
    issues = []
    for name in names:
        obj = package[name]
        if obj.is_alias:
            # Native dependency types (for example within.Preconditioner) have
            # no statically loaded target in this package-only signature audit.
            if not obj.target_path.startswith(f"{package.path}."):
                continue
            obj = obj.final_target
        if not obj.is_function or obj.docstring is None:
            continue
        sections = [
            section
            for section in obj.docstring.parsed
            if section.kind.value == "parameters"
        ]
        permitted = {
            parameter.name
            for parameter in obj.parameters
            if parameter.name not in {"self", "cls"}
        }
        required = {
            parameter.name
            for parameter in obj.parameters
            if parameter.name in permitted
            and parameter.kind
            not in {
                griffe.ParameterKind.var_keyword,
                griffe.ParameterKind.var_positional,
            }
        }
        documented = [
            parameter.name.lstrip("*")
            for section in sections
            for parameter in section.value
        ]
        missing = required - set(documented)
        repeated = [
            parameter for parameter, count in Counter(documented).items() if count > 1
        ]
        if missing:
            issues.append(f"{name}: undocumented parameters {sorted(missing)}")
        if repeated:
            issues.append(f"{name}: duplicate parameter entries {repeated}")
        accepts_kwargs = any(
            parameter.kind == griffe.ParameterKind.var_keyword
            for parameter in obj.parameters
        )
        if not accepts_kwargs:
            extras = set(documented) - permitted
            if extras:
                issues.append(
                    f"{name}: parameters absent from signature {sorted(extras)}"
                )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs"
    )
    parser.add_argument(
        "--site-dir", type=Path, help="Also validate rendered HTML links"
    )
    args = parser.parse_args()
    docs = args.docs_dir.resolve()
    config = yaml.safe_load((docs / "_quarto.yml").read_text())
    registered = registered_pages(config)
    issues = []
    folded = [path.casefold() for path, _ in registered]
    for path, count in Counter(folded).items():
        if count > 1:
            issues.append(
                f"Reference page collision on case-insensitive filesystems: {path}"
            )
    for path, _ in registered:
        if not (docs / "reference" / f"{path}.qmd").is_file():
            issues.append(f"Missing generated page: {path}; run pixi run docs-build")
    pages = sorted((docs / "reference").glob("*.qmd"))
    pages += sorted((docs / "reference-guide").glob("*.qmd"))
    issues.extend(check_links(docs, pages))
    package = griffe.load(
        "pyfixest",
        search_paths=[docs.parent],
        docstring_parser="numpy",
        allow_inspection=False,
    )
    issues.extend(
        check_parameters(package, [name for _, names in registered for name in names])
    )
    if args.site_dir:
        issues.extend(check_rendered_links(args.site_dir))
    if issues:
        raise SystemExit("\n".join(issues))
    print(
        f"Reference checks passed: {len(registered)} registered pages, {len(pages)} pages checked."
    )


if __name__ == "__main__":
    main()
