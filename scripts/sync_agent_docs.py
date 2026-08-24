"""Build version-matched agent documentation from rendered Quarto Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import SplitResult, unquote, urlsplit, urlunsplit

_MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\([^)]*\)")
_HTML_IMAGE_RE = re.compile(r"<img\b(?P<attrs>[^>]*)>", re.I)
_ALT_RE = re.compile(r"\balt=[\"'](?P<alt>[^\"']*)[\"']", re.I)
_MARKDOWN_LINK_RE = re.compile(
    r"(?<!!)\[(?P<label>[^\]]+)\]\((?P<target><[^>]+>|[^)\s]+)"
    r"(?P<title>\s+(?:\"[^\"]*\"|'[^']*'))?\)"
)
_HTML_LINK_RE = re.compile(
    r"(?P<prefix><a\s+[^>]*href=[\"'])(?P<target>[^\"']+)(?P<suffix>[\"'])",
    re.I,
)
_CARGO_VERSION_RE = re.compile(
    r"^\[package\]\s*$.*?^version\s*=\s*[\"'](?P<version>[^\"']+)[\"']",
    re.M | re.S,
)
_LOCAL_HOSTS = {"pyfixest.org", "www.pyfixest.org"}


class AgentDocsGenerationError(ValueError):
    """Report an invalid inventory, missing input, or generated-file drift."""


@dataclass(frozen=True, slots=True)
class PageSpec:
    """Describe one narrative page included in the package corpus."""

    source: PurePosixPath
    title: str
    description: str

    @property
    def rendered(self) -> PurePosixPath:
        """Return the page path emitted by Quarto's llms format."""
        return self.source.with_suffix(".llms.md")

    @property
    def bundled(self) -> PurePosixPath:
        """Return the page path relative to the package docs directory."""
        return PurePosixPath("pages") / self.source.with_suffix(".md")


@dataclass(frozen=True, slots=True)
class Inventory:
    """Hold validated agent-documentation configuration."""

    site_url: str
    pages: tuple[PageSpec, ...]


def _load_inventory(path: Path) -> Inventory:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentDocsGenerationError(f"Cannot read inventory {path}: {exc}") from exc

    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise AgentDocsGenerationError("Agent docs inventory needs schema_version 1.")
    site_url = raw.get("site_url")
    raw_pages = raw.get("pages")
    if not isinstance(site_url, str) or not site_url.startswith("https://"):
        raise AgentDocsGenerationError("Inventory site_url must be an HTTPS URL.")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise AgentDocsGenerationError("Inventory pages must be a non-empty list.")

    pages: list[PageSpec] = []
    for position, raw_page in enumerate(raw_pages, start=1):
        if not isinstance(raw_page, dict):
            raise AgentDocsGenerationError(
                f"Inventory page {position} must be an object."
            )
        values = [raw_page.get(key) for key in ("source", "title", "description")]
        if not all(isinstance(value, str) and value for value in values):
            raise AgentDocsGenerationError(
                f"Inventory page {position} needs source, title, and description."
            )
        source = PurePosixPath(values[0])
        if source.is_absolute() or ".." in source.parts:
            raise AgentDocsGenerationError(f"Unsafe inventory source: {source}")
        pages.append(PageSpec(source, values[1], values[2]))

    sources = [page.source for page in pages]
    if len(set(sources)) != len(sources):
        raise AgentDocsGenerationError("Inventory page sources must be unique.")
    return Inventory(site_url.rstrip("/") + "/", tuple(pages))


def _cargo_version(path: Path) -> str:
    match = _CARGO_VERSION_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise AgentDocsGenerationError(f"Cannot find package version in {path}.")
    return match.group("version")


def _image_text(alt: str) -> str:
    return f"Image: {alt}" if alt.strip() else "Image omitted."


def _replace_images(text: str) -> str:
    text = _MARKDOWN_IMAGE_RE.sub(lambda match: _image_text(match.group("alt")), text)

    def replace_html(match: re.Match[str]) -> str:
        alt_match = _ALT_RE.search(match.group("attrs"))
        return _image_text(alt_match.group("alt") if alt_match else "")

    return _HTML_IMAGE_RE.sub(replace_html, text)


def _as_rendered_path(path: PurePosixPath) -> PurePosixPath:
    value = path.as_posix()
    if value.endswith(".llms.md"):
        return path
    if path.suffix in {".html", ".qmd", ".md"}:
        return path.with_suffix(".llms.md")
    if path.suffix == "":
        return path / "index.llms.md"
    return path


def _local_rendered_target(
    target: str, source: PurePosixPath
) -> tuple[PurePosixPath, SplitResult] | None:
    parsed = urlsplit(target.strip("<>"))
    if parsed.scheme not in {"", "http", "https"}:
        return None
    if parsed.netloc and parsed.netloc.casefold() not in _LOCAL_HOSTS:
        return None
    if not parsed.path or ("@" in parsed.path and "/" not in parsed.path):
        return None
    decoded = unquote(parsed.path)
    if decoded.startswith("/"):
        normalized = posixpath.normpath(decoded.lstrip("/"))
    else:
        normalized = posixpath.normpath(str(source.parent / decoded))
    return _as_rendered_path(PurePosixPath(normalized)), parsed


def _rewrite_target(
    *,
    target: str,
    source: PageSpec,
    bundled_pages: dict[PurePosixPath, PurePosixPath],
    site_url: str,
) -> str:
    local = _local_rendered_target(target, source.rendered)
    if local is None:
        return target
    rendered_target, parsed = local
    bundled_target = bundled_pages.get(rendered_target)
    if bundled_target is None:
        path = site_url + rendered_target.as_posix()
    else:
        path = posixpath.relpath(
            bundled_target.as_posix(), start=source.bundled.parent.as_posix()
        )
    return urlunsplit(("", "", path, parsed.query, parsed.fragment))


def _rewrite_page(*, text: str, page: PageSpec, inventory: Inventory) -> str:
    bundled_pages = {
        candidate.rendered: candidate.bundled for candidate in inventory.pages
    }
    text = _replace_images(text)

    def replace_markdown(match: re.Match[str]) -> str:
        rewritten = _rewrite_target(
            target=match.group("target").strip("<>"),
            source=page,
            bundled_pages=bundled_pages,
            site_url=inventory.site_url,
        )
        return f"[{match.group('label')}]({rewritten}{match.group('title') or ''})"

    text = _MARKDOWN_LINK_RE.sub(replace_markdown, text)

    def replace_html(match: re.Match[str]) -> str:
        rewritten = _rewrite_target(
            target=match.group("target"),
            source=page,
            bundled_pages=bundled_pages,
            site_url=inventory.site_url,
        )
        return match.group("prefix") + rewritten + match.group("suffix")

    text = _HTML_LINK_RE.sub(replace_html, text)
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"


def _index_text(inventory: Inventory, version: str) -> str:
    lines = [
        "# PyFixest documentation for agents",
        "",
        f"This corpus is bundled with PyFixest {version} and matches that release.",
        "For API entries not bundled here, follow links to the rendered website.",
        "",
        "## Narrative pages",
        "",
    ]
    for page in inventory.pages:
        lines.extend(
            [
                f"- [{page.title}]({page.bundled.as_posix()})",
                f"  {page.description}",
            ]
        )
    return "\n".join(lines) + "\n"


def _llms_text(inventory: Inventory, version: str) -> str:
    lines = [
        "# PyFixest",
        "",
        f"> Version-matched narrative documentation for PyFixest {version}.",
        "",
        "## Pages",
        "",
    ]
    lines.extend(
        f"- [{page.title}]({page.bundled.as_posix()}): {page.description}"
        for page in inventory.pages
    )
    return "\n".join(lines) + "\n"


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _expected_files(
    *, site: Path, inventory: Inventory, package_version: str
) -> dict[PurePosixPath, str]:
    files: dict[PurePosixPath, str] = {}
    for page in inventory.pages:
        rendered = site / page.rendered
        if not rendered.is_file():
            raise AgentDocsGenerationError(f"Missing rendered page: {rendered}")
        files[page.bundled] = _rewrite_page(
            text=rendered.read_text(encoding="utf-8"),
            page=page,
            inventory=inventory,
        )

    files[PurePosixPath("index.md")] = _index_text(inventory, package_version)
    files[PurePosixPath("llms.txt")] = _llms_text(inventory, package_version)
    manifest_files = [
        {
            "path": path.as_posix(),
            "sha256": _sha256(content),
            "bytes": len(content.encode()),
        }
        for path, content in sorted(files.items())
    ]
    manifest = {
        "schema_version": 1,
        "package_version": package_version,
        "page_count": len(inventory.pages),
        "files": manifest_files,
    }
    files[PurePosixPath("manifest.json")] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return files


def generate_bundle(
    *,
    site: Path,
    inventory_path: Path,
    output: Path,
    cargo_manifest: Path,
    check: bool = False,
) -> None:
    """Generate the package corpus or fail if checked files have drifted."""
    inventory = _load_inventory(inventory_path)
    expected = _expected_files(
        site=site,
        inventory=inventory,
        package_version=_cargo_version(cargo_manifest),
    )
    actual_paths = (
        {
            PurePosixPath(path.relative_to(output).as_posix())
            for path in output.rglob("*")
            if path.is_file()
        }
        if output.is_dir()
        else set()
    )

    if check:
        drift = [
            path.as_posix()
            for path, content in expected.items()
            if not (output / path).is_file()
            or (output / path).read_text(encoding="utf-8") != content
        ]
        unexpected = sorted(path.as_posix() for path in actual_paths - expected.keys())
        if drift or unexpected:
            details = [*(f"out of date: {path}" for path in drift)]
            details.extend(f"unexpected: {path}" for path in unexpected)
            raise AgentDocsGenerationError(
                "Generated docs differ:\n" + "\n".join(details)
            )
        return

    for stale in actual_paths - expected.keys():
        (output / stale).unlink()
    for path, content in expected.items():
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


def main() -> int:
    """Run the documentation synchronizer from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("docs/_site"))
    parser.add_argument("--inventory", type=Path, default=Path("docs/agent-docs.json"))
    parser.add_argument("--output", type=Path, default=Path("pyfixest/docs"))
    parser.add_argument("--cargo-manifest", type=Path, default=Path("Cargo.toml"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        generate_bundle(
            site=args.site,
            inventory_path=args.inventory,
            output=args.output,
            cargo_manifest=args.cargo_manifest,
            check=args.check,
        )
    except AgentDocsGenerationError as exc:
        parser.exit(1, f"agent docs generation failed:\n{exc}\n")
    action = "matches generated output" if args.check else "generated"
    print(f"agent docs {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
