"""Contract tests for the statically documented top-level API."""

from __future__ import annotations

import inspect
from pathlib import Path

import pyfixest as pf

# These exports are intentionally public but do not yet have standalone reference
# pages. Keeping the list explicit prevents accidental API growth; an export must be
# registered in quartodoc or added here with a conscious review decision.
_REFERENCE_ALLOWLIST = {
    "Preconditioner",
    "SaturatedEventStudy",
    "get_bartik_data",
    "get_data",
    "get_encouragement_data",
    "get_ivf_data",
    "get_motherhood_event_study_data",
    "get_twin_data",
    "get_worker_panel",
    "qplot",
}


def _quartodoc_leaf_names() -> set[str]:
    config = Path(__file__).parents[1] / "docs" / "_quarto.yml"
    quartodoc_config = config.read_text(encoding="utf-8").split("quartodoc:", 1)[1]
    return {
        line.removeprefix("-").strip().rsplit(".", 1)[-1]
        for raw_line in quartodoc_config.splitlines()
        if (line := raw_line.strip()).startswith("-")
        and "/" not in line
        and not line.startswith("#")
    }


def test_all_top_level_exports_resolve() -> None:
    unresolved = {
        name: repr(exc)
        for name in pf.__all__
        if (exc := _resolve_error(name)) is not None
    }

    assert unresolved == {}


def _resolve_error(name: str) -> Exception | None:
    try:
        getattr(pf, name)
    except Exception as exc:  # pragma: no cover - assertion reports the concrete error
        return exc
    return None


def test_dir_includes_public_exports_and_module_metadata() -> None:
    names = set(dir(pf))

    assert set(pf.__all__) <= names
    assert {"__doc__", "__name__", "__package__", "__spec__", "__version__"} <= names


def test_public_callables_have_reference_pages_or_are_allowlisted() -> None:
    public_callables = {
        name
        for name in pf.__all__
        if inspect.isroutine(getattr(pf, name)) or inspect.isclass(getattr(pf, name))
    }
    registered = _quartodoc_leaf_names()

    assert public_callables >= _REFERENCE_ALLOWLIST
    assert _REFERENCE_ALLOWLIST.isdisjoint(registered)
    assert public_callables <= registered | _REFERENCE_ALLOWLIST
