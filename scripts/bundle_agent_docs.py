"""Bundle Quarto's machine-readable documentation into the Python package."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import tomllib

_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)(\{[^}]*\})?")
_BADGE_LINK_RE = re.compile(
    r"\[!\[[^\]]*\]\([^)]*\)(?:\{[^}]*\})?\]\([^)]*\)(?:\{[^}]*\})?"
)
_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_HTML_BADGE_RE = re.compile(r"<a\b[^>]*>\s*<img\b[^>]*>\s*</a>", re.IGNORECASE)
_HTML_ALT_RE = re.compile(r"\balt=(['\"])(.*?)\1", re.IGNORECASE)
_FIG_ALT_RE = re.compile(r"\bfig-alt=(['\"])(.*?)\1", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_STUB = "index.llms.md"


class AgentDocsBundleError(ValueError):
    """Report missing or unusable Quarto output."""


def _image_text(alt: str) -> str:
    return f"[Figure: {alt}]" if alt else "[Figure]"


def _replace_markdown_image(match: re.Match[str]) -> str:
    fig_alt = _FIG_ALT_RE.search(match[2]) if match[2] else None
    return _image_text(match[1] or (fig_alt[2] if fig_alt else ""))


def _replace_html_image(match: re.Match[str]) -> str:
    alt = _HTML_ALT_RE.search(match[0])
    return _image_text(alt[2] if alt else "")


def _normalize_markdown(text: str) -> str:
    """Replace images with alt-text placeholders, leaving code blocks literal."""
    normalized: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        match = _FENCE_RE.match(line)
        if match:
            marker = match[1]
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
        elif fence is None:
            line = _HTML_BADGE_RE.sub("", line)
            line = _BADGE_LINK_RE.sub("", line)
            line = _MARKDOWN_IMAGE_RE.sub(_replace_markdown_image, line)
            line = _HTML_IMAGE_RE.sub(_replace_html_image, line)
        normalized.append(line)
    return "".join(normalized)


def _index_text(site: Path, version_file: Path) -> str:
    """Drop the redirect stub from llms.txt and stamp the package version."""
    manifest = tomllib.loads(version_file.read_text(encoding="utf-8"))
    lines = (site / "llms.txt").read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if f"]({_STUB})" not in line]
    after = next((i for i, line in enumerate(kept) if line.startswith(">")), 0) + 1
    while after < len(kept) and kept[after].startswith(">"):
        after += 1
    kept[after:after] = ["", f"Version: {manifest['package']['version']}"]
    return "\n".join(kept) + "\n"


def bundle(*, site: Path, output: Path, version_file: Path) -> int:
    """Copy Quarto's llms output into the package and return the page count."""
    if not (site / "llms.txt").is_file():
        raise AgentDocsBundleError(f"Missing Quarto llms index: {site / 'llms.txt'}")
    found = sorted(site.rglob("*.llms.md"))
    pages = [page for page in found if page.relative_to(site).as_posix() != _STUB]
    if not pages:
        raise AgentDocsBundleError(f"No Quarto .llms.md pages found in {site}")

    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    index = _index_text(site, version_file)
    (output / "llms.txt").write_text(index, encoding="utf-8")
    for page in pages:
        destination = output / page.relative_to(site)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = _normalize_markdown(page.read_text(encoding="utf-8"))
        destination.write_text(text, encoding="utf-8")
    return len(pages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("docs/_site"))
    parser.add_argument("--output", type=Path, default=Path("pyfixest/docs"))
    parser.add_argument("--version-file", type=Path, default=Path("Cargo.toml"))
    args = parser.parse_args()
    try:
        pages = bundle(
            site=args.site, output=args.output, version_file=args.version_file
        )
    except AgentDocsBundleError as exc:
        parser.exit(1, f"agent docs bundling failed:\n{exc}\n")
    print(f"bundled {pages} documentation pages into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
