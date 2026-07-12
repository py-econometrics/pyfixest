"""Unit tests for the deterministic release-wheel asset checker."""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_wheel_contents.py"


@pytest.fixture(scope="module")
def wheel_checker():
    spec = importlib.util.spec_from_file_location("check_wheel_contents", CHECKER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_wheel(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _valid_members() -> dict[str, str]:
    prefix = "pyfixest/docs/"
    members = {
        f"{prefix}index.md": "# Docs\n",
        f"{prefix}llms.txt": "# PyFixest\n",
        "skills/pyfixest/SKILL.md": "---\nname: pyfixest\ndescription: Test.\n---\n",
        "skills/pyfixest/agents/openai.yaml": "interface: {}\n",
    }
    pages = []
    for number in range(7):
        members[f"skills/pyfixest/references/reference-{number}.md"] = "# Ref\n"
    members[f"{prefix}pages/guide.md"] = "# Guide\n"
    pages.append({"route": "guide"})
    members[f"{prefix}manifest.json"] = json.dumps({"pages": pages})
    return members


def test_valid_wheel_passes(wheel_checker, tmp_path):
    wheel = tmp_path / "pyfixest-0.0.0-py3-none-any.whl"
    _write_wheel(wheel, _valid_members())

    wheel_checker.check_wheel(wheel)


def test_forbidden_asset_fails(wheel_checker, tmp_path):
    wheel = tmp_path / "pyfixest-0.0.0-py3-none-any.whl"
    members = _valid_members()
    members["pyfixest/docs/plot.png"] = "not really an image"
    _write_wheel(wheel, members)

    with pytest.raises(wheel_checker.WheelContentsError, match="forbidden assets"):
        wheel_checker.check_wheel(wheel)


def test_missing_manifest_page_fails(wheel_checker, tmp_path):
    wheel = tmp_path / "pyfixest-0.0.0-py3-none-any.whl"
    members = _valid_members()
    members["pyfixest/docs/manifest.json"] = json.dumps(
        {"pages": [{"route": "missing"}]}
    )
    _write_wheel(wheel, members)

    with pytest.raises(wheel_checker.WheelContentsError, match="manifest page missing"):
        wheel_checker.check_wheel(wheel)
