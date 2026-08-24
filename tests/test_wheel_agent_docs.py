from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.check_wheel_contents import WheelDocsError, check_wheel


def _write_test_wheel(
    tmp_path: Path, *, add_binary: bool = False, corrupt_index: bool = False
) -> Path:
    wheel = tmp_path / "pyfixest-0.60.0-py3-none-any.whl"
    manifest = json.loads(
        Path("pyfixest/docs/manifest.json").read_text(encoding="utf-8")
    )
    package_sources = {
        source
        for case in json.loads(
            Path("docs/agent-search-cases.json").read_text(encoding="utf-8")
        )
        for source in case["package_sources"]
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "pyfixest-0.60.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: pyfixest\nVersion: 0.60.0\n",
        )
        archive.write("pyfixest/docs/manifest.json", "pyfixest/docs/manifest.json")
        for entry in manifest["files"]:
            source = Path("pyfixest/docs") / entry["path"]
            if corrupt_index and entry["path"] == "index.md":
                archive.writestr(source.as_posix(), "changed\n")
            else:
                archive.write(source, source.as_posix())
        for source in package_sources:
            if not source.startswith("pyfixest/docs/"):
                archive.write(source, source)
        if add_binary:
            archive.writestr("pyfixest/docs/plot.png", b"not an image")
    return wheel


def test_wheel_agent_docs_are_complete(tmp_path: Path) -> None:
    wheel = _write_test_wheel(tmp_path)

    check_wheel(wheel=wheel, cases_path=Path("docs/agent-search-cases.json"))


def test_wheel_agent_docs_reject_binary_assets(tmp_path: Path) -> None:
    wheel = _write_test_wheel(tmp_path, add_binary=True)

    with pytest.raises(WheelDocsError, match="Forbidden docs asset"):
        check_wheel(wheel=wheel, cases_path=Path("docs/agent-search-cases.json"))


def test_wheel_agent_docs_reject_checksum_mismatch(tmp_path: Path) -> None:
    wheel = _write_test_wheel(tmp_path, corrupt_index=True)

    with pytest.raises(WheelDocsError, match="checksum mismatch"):
        check_wheel(wheel=wheel, cases_path=Path("docs/agent-search-cases.json"))
