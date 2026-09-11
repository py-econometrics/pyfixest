"""Check the rendered llms.txt index and the internal links in its pages."""

from __future__ import annotations

import argparse
import posixpath
import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

_MARKDOWN_LINK_RE = re.compile(
    r"(?<!!)\[[^\]]+\]\((?P<target><[^>]+>|[^)\s]+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'))?\)"
)
_HTML_LINK_RE = re.compile(r"<a\s+[^>]*href=[\"'](?P<target>[^\"']+)[\"']", re.I)


class AgentDocsError(ValueError):
    """Report one or more violations of the rendered documentation contract."""


def _markdown_targets(text: str) -> list[str]:
    matches = [*_MARKDOWN_LINK_RE.finditer(text), *_HTML_LINK_RE.finditer(text)]
    return [match.group("target").strip("<>") for match in matches]


def _local_path(target: str, source: PurePosixPath) -> PurePosixPath | None:
    """Resolve a link target against its page, or return None if it is not local."""
    parsed = urlsplit(target)
    is_email = "@" in parsed.path and "/" not in parsed.path
    if parsed.scheme or parsed.netloc or not parsed.path or is_email:
        return None
    decoded = unquote(parsed.path)
    if not decoded.startswith("/"):
        decoded = str(source.parent / decoded)
    normalized = posixpath.normpath(decoded.lstrip("/"))
    path = PurePosixPath("index.html" if normalized == "." else normalized)
    if path.is_absolute() or ".." in path.parts:
        raise AgentDocsError(f"Local link escapes the rendered site: {target}")
    return path


def _path_exists(site: Path, target: PurePosixPath) -> bool:
    root = site.resolve()
    candidates = (target, target.with_suffix(".html"), target / "index.html")
    resolved = [(root / candidate).resolve() for candidate in candidates]
    return any(path.is_relative_to(root) and path.is_file() for path in resolved)


def _check_links(site: Path, source: PurePosixPath, errors: list[str]) -> set[str]:
    """Resolve every link on one page, recording the targets that do not exist."""
    targets: set[str] = set()
    for target in _markdown_targets((site / source).read_text(encoding="utf-8")):
        try:
            path = _local_path(target, source)
        except AgentDocsError as exc:
            errors.append(f"{exc} (in {source})")
            continue
        if path is not None:
            targets.add(path.as_posix())
            if not _path_exists(site, path):
                errors.append(f"Broken internal link in {source}: {target}")
    return targets


def check_agent_docs(*, site: Path) -> None:
    """Check that llms.txt indexes every rendered page and that its links resolve."""
    errors: list[str] = []
    index = PurePosixPath("llms.txt")
    if not (site / index).is_file():
        raise AgentDocsError(f"Missing rendered index: {site / index}")

    indexed = {t for t in _check_links(site, index, errors) if t.endswith(".llms.md")}
    if not indexed:
        errors.append("llms.txt does not index any .llms.md pages.")
    rendered = {path.relative_to(site).as_posix() for path in site.rglob("*.llms.md")}
    for page in rendered - indexed:
        errors.append(f"Rendered page is missing from llms.txt: {page}")
    for page in indexed - rendered:
        errors.append(f"llms.txt indexes a missing page: {page}")
    for page in sorted(indexed & rendered):
        _check_links(site, PurePosixPath(page), errors)
    if errors:
        raise AgentDocsError("\n".join(sorted(errors)))


def main() -> int:
    """Run the rendered-documentation checks from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("docs/_site"))
    try:
        check_agent_docs(site=parser.parse_args().site)
    except AgentDocsError as exc:
        parser.exit(1, f"agent docs check failed:\n{exc}\n")
    print("agent docs check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
