"""Validate agent-readable documentation inside a built PyFixest wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

_DOCS_PREFIX = PurePosixPath("pyfixest/docs")
_MAX_DOCS_BYTES = 1024 * 1024
_REQUIRED_DOCS = {
    (_DOCS_PREFIX / "index.md").as_posix(),
    (_DOCS_PREFIX / "llms.txt").as_posix(),
    (_DOCS_PREFIX / "manifest.json").as_posix(),
}
_FORBIDDEN_SUFFIXES = {
    ".gif",
    ".html",
    ".ipynb",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".svg",
    ".webp",
}
_FORBIDDEN_PARTS = {"_freeze", "_site", "llms-full.txt"}
_MARKDOWN_LINK_RE = re.compile(
    r"(?<!!)\[[^\]]+\]\((?P<target><[^>]+>|[^)\s]+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'))?\)"
)
_HTML_LINK_RE = re.compile(r"<a\s+[^>]*href=[\"'](?P<target>[^\"']+)[\"']", re.I)


class WheelDocsError(ValueError):
    """Report invalid or missing documentation in a wheel."""


def _find_wheel(path: Path) -> Path:
    if path.is_file() and path.suffix == ".whl":
        return path
    wheels = sorted(path.glob("*.whl")) if path.is_dir() else []
    if len(wheels) != 1:
        raise WheelDocsError(f"Expected one wheel in {path}, found {len(wheels)}.")
    return wheels[0]


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _link_targets(text: str) -> list[str]:
    targets = [match.group("target") for match in _MARKDOWN_LINK_RE.finditer(text)]
    targets.extend(match.group("target") for match in _HTML_LINK_RE.finditer(text))
    return [target.strip("<>") for target in targets]


def _check_manifest(
    archive: zipfile.ZipFile, names: set[str], package_version: str
) -> dict[str, object]:
    manifest_name = (_DOCS_PREFIX / "manifest.json").as_posix()
    try:
        manifest = json.loads(archive.read(manifest_name))
    except (KeyError, json.JSONDecodeError) as exc:
        raise WheelDocsError("Missing or malformed docs manifest.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise WheelDocsError("Docs manifest needs schema_version 1.")
    if manifest.get("package_version") != package_version:
        raise WheelDocsError(
            "Docs version does not match wheel metadata: "
            f"{manifest.get('package_version')!r} != {package_version!r}."
        )
    if not isinstance(manifest.get("page_count"), int) or manifest["page_count"] < 20:
        raise WheelDocsError("Docs manifest must contain at least 20 narrative pages.")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise WheelDocsError("Docs manifest has no files.")

    declared: set[str] = set()
    page_count = 0
    for raw_file in raw_files:
        if not isinstance(raw_file, dict):
            raise WheelDocsError("Docs manifest file entries must be objects.")
        relative = raw_file.get("path")
        checksum = raw_file.get("sha256")
        byte_count = raw_file.get("bytes")
        if not isinstance(relative, str) or not isinstance(checksum, str):
            raise WheelDocsError("Docs manifest file entry is malformed.")
        member = (_DOCS_PREFIX / relative).as_posix()
        if member in declared:
            raise WheelDocsError(f"Duplicate docs manifest path: {relative}")
        declared.add(member)
        if member not in names:
            raise WheelDocsError(f"Docs manifest file is missing: {relative}")
        content = archive.read(member)
        if hashlib.sha256(content).hexdigest() != checksum:
            raise WheelDocsError(f"Docs manifest checksum mismatch: {relative}")
        if len(content) != byte_count:
            raise WheelDocsError(f"Docs manifest byte count mismatch: {relative}")
        if relative.startswith("pages/"):
            page_count += 1

    if page_count != manifest["page_count"]:
        raise WheelDocsError("Docs manifest page_count does not match its files.")
    actual = {name for name in names if name.startswith(_DOCS_PREFIX.as_posix() + "/")}
    allowed = declared | {manifest_name}
    if actual != allowed:
        extras = sorted(actual - allowed)
        missing = sorted(allowed - actual)
        raise WheelDocsError(
            f"Docs manifest inventory mismatch; extra={extras}, missing={missing}."
        )
    return manifest


def _wheel_version(archive: zipfile.ZipFile, names: set[str]) -> str:
    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    if len(metadata_names) != 1:
        raise WheelDocsError("Wheel must contain exactly one METADATA file.")
    metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
    version = metadata.get("Version")
    if version is None:
        raise WheelDocsError("Wheel METADATA has no Version field.")
    return version


def _check_docs_files(archive: zipfile.ZipFile, names: set[str]) -> None:
    missing = sorted(_REQUIRED_DOCS - names)
    if missing:
        raise WheelDocsError(f"Wheel is missing required docs files: {missing}.")
    doc_names = sorted(
        name for name in names if name.startswith(_DOCS_PREFIX.as_posix() + "/")
    )
    total_size = sum(archive.getinfo(name).file_size for name in doc_names)
    if total_size > _MAX_DOCS_BYTES:
        raise WheelDocsError(
            f"Docs corpus is {total_size} bytes; maximum is {_MAX_DOCS_BYTES}."
        )
    for name in doc_names:
        path = PurePosixPath(name)
        if path.suffix.casefold() in _FORBIDDEN_SUFFIXES or any(
            part in _FORBIDDEN_PARTS for part in path.parts
        ):
            raise WheelDocsError(f"Forbidden docs asset in wheel: {name}")


def _check_internal_links(archive: zipfile.ZipFile, names: set[str]) -> None:
    text_names = sorted(
        name
        for name in names
        if name.startswith(_DOCS_PREFIX.as_posix() + "/")
        and PurePosixPath(name).suffix in {".md", ".txt"}
    )
    for name in text_names:
        text = archive.read(name).decode()
        source = PurePosixPath(name)
        for target in _link_targets(text):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            if "@" in parsed.path and "/" not in parsed.path:
                continue
            resolved = PurePosixPath(
                posixpath.normpath(str(source.parent / unquote(parsed.path)))
            ).as_posix()
            if resolved not in names:
                raise WheelDocsError(f"Broken bundled link in {name}: {target}")


def _check_retrieval_cases(
    archive: zipfile.ZipFile, names: set[str], cases_path: Path
) -> None:
    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    for raw_case in raw_cases:
        case_id = raw_case["id"]
        sources = raw_case.get("package_sources")
        terms = raw_case["terms"]
        if not isinstance(sources, list) or not sources:
            raise WheelDocsError(f"Retrieval case {case_id!r} has no package_sources.")
        missing_sources = [source for source in sources if source not in names]
        if missing_sources:
            raise WheelDocsError(
                f"Retrieval case {case_id!r} misses package sources: {missing_sources}."
            )
        corpus = _normalized_text(
            "\n".join(archive.read(source).decode() for source in sources)
        )
        missing_terms = [term for term in terms if _normalized_text(term) not in corpus]
        if missing_terms:
            raise WheelDocsError(
                f"Retrieval case {case_id!r} misses terms: {missing_terms}."
            )


def check_wheel(*, wheel: Path, cases_path: Path) -> None:
    """Validate package docs, metadata, links, and retrieval cases in a wheel."""
    wheel = _find_wheel(wheel)
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        package_version = _wheel_version(archive, names)
        _check_docs_files(archive, names)
        _check_manifest(archive, names, package_version)
        _check_internal_links(archive, names)
        _check_retrieval_cases(archive, names, cases_path)


def main() -> int:
    """Run wheel-content validation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wheel", type=Path)
    source.add_argument("--wheel-dir", type=Path)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("docs/agent-search-cases.json"),
    )
    args = parser.parse_args()
    try:
        check_wheel(wheel=args.wheel or args.wheel_dir, cases_path=args.cases)
    except (OSError, WheelDocsError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"wheel docs check failed:\n{exc}\n")
    print("wheel docs check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
