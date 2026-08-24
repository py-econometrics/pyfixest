"""Validate rendered documentation intended for language-model retrieval."""

from __future__ import annotations

import argparse
import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

_MARKDOWN_LINK_RE = re.compile(
    r"(?<!!)\[[^\]]+\]\((?P<target><[^>]+>|[^)\s]+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'))?\)"
)
_HTML_LINK_RE = re.compile(r"<a\s+[^>]*href=[\"'](?P<target>[^\"']+)[\"']", re.I)
_QUARTO_EXCLUDE_RE = re.compile(r"^\s*-\s*[\"']!(?P<path>.+?)[\"']\s*$")
_LOCAL_HOSTS = {"pyfixest.org", "www.pyfixest.org"}


class AgentDocsError(ValueError):
    """Report one or more violations of the rendered documentation contract."""


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    """Define terms that must occur in a set of authoritative pages."""

    case_id: str
    question: str
    authoritative_sources: tuple[PurePosixPath, ...]
    terms: tuple[str, ...]


def _markdown_targets(text: str) -> list[str]:
    targets = [match.group("target") for match in _MARKDOWN_LINK_RE.finditer(text)]
    targets.extend(match.group("target") for match in _HTML_LINK_RE.finditer(text))
    return [target.strip("<>") for target in targets]


def _load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentDocsError(f"Cannot read retrieval cases from {path}: {exc}") from exc

    if not isinstance(raw_cases, list) or not raw_cases:
        raise AgentDocsError("Retrieval cases must be a non-empty JSON list.")

    cases: list[RetrievalCase] = []
    for position, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise AgentDocsError(f"Retrieval case {position} must be an object.")
        try:
            case_id = raw_case["id"]
            question = raw_case["question"]
            sources = raw_case["authoritative_sources"]
            terms = raw_case["terms"]
        except KeyError as exc:
            raise AgentDocsError(
                f"Retrieval case {position} is missing {exc.args[0]!r}."
            ) from exc
        if not isinstance(case_id, str) or not case_id:
            raise AgentDocsError(f"Retrieval case {position} has an invalid id.")
        if not isinstance(question, str) or not question:
            raise AgentDocsError(f"Retrieval case {case_id!r} has no question.")
        if (
            not isinstance(sources, list)
            or not sources
            or not all(
                isinstance(source, str) and source.endswith(".llms.md")
                for source in sources
            )
        ):
            raise AgentDocsError(
                f"Retrieval case {case_id!r} needs .llms.md authoritative sources."
            )
        if (
            not isinstance(terms, list)
            or not terms
            or not all(isinstance(term, str) and term for term in terms)
        ):
            raise AgentDocsError(f"Retrieval case {case_id!r} needs search terms.")
        cases.append(
            RetrievalCase(
                case_id=case_id,
                question=question,
                authoritative_sources=tuple(
                    PurePosixPath(source) for source in sources
                ),
                terms=tuple(terms),
            )
        )
    return cases


def _excluded_outputs(quarto_config: Path | None) -> set[PurePosixPath]:
    if quarto_config is None:
        return set()
    outputs: set[PurePosixPath] = set()
    for line in quarto_config.read_text(encoding="utf-8").splitlines():
        match = _QUARTO_EXCLUDE_RE.match(line)
        if match is None:
            continue
        source = PurePosixPath(match.group("path"))
        outputs.add(source.with_suffix(".html"))
        outputs.add(source.with_suffix(".llms.md"))
    return outputs


def _local_path(target: str, source: PurePosixPath) -> PurePosixPath | None:
    parsed = urlsplit(target)
    if parsed.scheme not in {"", "http", "https"}:
        return None
    if parsed.netloc and parsed.netloc.casefold() not in _LOCAL_HOSTS:
        return None
    if "@" in parsed.path and "/" not in parsed.path:
        return None
    if not parsed.path:
        return None

    decoded = unquote(parsed.path)
    if decoded.startswith("/"):
        normalized = posixpath.normpath(decoded.lstrip("/"))
    else:
        normalized = posixpath.normpath(str(source.parent / decoded))
    if normalized == ".":
        return PurePosixPath("index.html")
    return PurePosixPath(normalized)


def _path_exists(site: Path, target: PurePosixPath) -> bool:
    candidates = [target]
    if target.as_posix().endswith("/"):
        candidates.append(target / "index.html")
    if target.suffix == "":
        candidates.extend([target.with_suffix(".html"), target / "index.html"])
    return any((site / candidate).is_file() for candidate in candidates)


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def check_agent_docs(
    *, site: Path, cases_path: Path, quarto_config: Path | None = None
) -> None:
    """Check the rendered index, linked pages, links, and retrieval cases."""
    errors: list[str] = []
    index_path = site / "llms.txt"
    if not index_path.is_file():
        raise AgentDocsError(f"Missing rendered index: {index_path}")

    index_text = index_path.read_text(encoding="utf-8")
    index_source = PurePosixPath("llms.txt")
    indexed_pages = {
        path
        for target in _markdown_targets(index_text)
        if (path := _local_path(target, index_source)) is not None
        and path.as_posix().endswith(".llms.md")
    }
    if not indexed_pages:
        errors.append("llms.txt does not index any .llms.md pages.")

    for page in sorted(indexed_pages):
        if not (site / page).is_file():
            errors.append(f"llms.txt indexes a missing page: {page}")

    excluded = _excluded_outputs(quarto_config)
    linked_targets: list[tuple[PurePosixPath, PurePosixPath]] = []
    documents = [(index_source, index_text)]
    documents.extend(
        (page, (site / page).read_text(encoding="utf-8"))
        for page in sorted(indexed_pages)
        if (site / page).is_file()
    )
    for source, text in documents:
        for raw_target in _markdown_targets(text):
            target = _local_path(raw_target, source)
            if target is None:
                continue
            linked_targets.append((source, target))
            if not _path_exists(site, target):
                errors.append(f"Broken internal link in {source}: {raw_target}")

    for source, target in linked_targets:
        if target in excluded:
            errors.append(f"Excluded draft linked from {source}: {target}")

    cases = _load_retrieval_cases(cases_path)
    for case in cases:
        source_text: list[str] = []
        for source in case.authoritative_sources:
            if source not in indexed_pages:
                errors.append(
                    f"Retrieval case {case.case_id!r} source is not indexed: {source}"
                )
                continue
            source_path = site / source
            if not source_path.is_file():
                errors.append(
                    f"Retrieval case {case.case_id!r} source is missing: {source}"
                )
                continue
            source_text.append(source_path.read_text(encoding="utf-8"))
        corpus = _normalized_text("\n".join(source_text))
        missing_terms = [
            term for term in case.terms if _normalized_text(term) not in corpus
        ]
        if missing_terms:
            errors.append(
                f"Retrieval case {case.case_id!r} misses terms: "
                + ", ".join(repr(term) for term in missing_terms)
            )

    if errors:
        raise AgentDocsError("\n".join(errors))


def main() -> int:
    """Run the agent-documentation checks from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("docs/_site"))
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("docs/agent-search-cases.json"),
    )
    parser.add_argument("--quarto-config", type=Path, default=Path("docs/_quarto.yml"))
    args = parser.parse_args()
    try:
        check_agent_docs(
            site=args.site,
            cases_path=args.cases,
            quarto_config=args.quarto_config,
        )
    except AgentDocsError as exc:
        parser.exit(1, f"agent docs check failed:\n{exc}\n")
    print("agent docs check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
