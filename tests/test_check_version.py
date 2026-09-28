"""Tests for release version validation and docs deployment selection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Import the module from .github/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".github"))
from check_version import (  # noqa: E402
    cargo_to_python_version,
    get_cargo_version,
)


class TestCargoToPythonVersion:
    """Test Cargo SemVer to PEP 440 conversion."""

    def test_stable_version(self):
        assert cargo_to_python_version("0.50.0") == "0.50.0"

    def test_alpha_version(self):
        assert cargo_to_python_version("0.50.0-alpha.1") == "0.50.0a1"

    def test_beta_version(self):
        assert cargo_to_python_version("0.50.0-beta.2") == "0.50.0b2"

    def test_major_version(self):
        assert cargo_to_python_version("1.0.0") == "1.0.0"

    def test_alpha_high_number(self):
        assert cargo_to_python_version("2.0.0-alpha.13") == "2.0.0a13"


class TestGetCargoVersion:
    """Test that get_cargo_version reads from the real Cargo.toml."""

    def test_returns_string(self):
        version = get_cargo_version()
        assert isinstance(version, str)
        assert len(version.split(".")) >= 2

    def test_matches_cargo_toml(self):
        cargo_toml = (ROOT / "Cargo.toml").read_text()
        version = get_cargo_version()
        assert f'version = "{version}"' in cargo_toml


class TestMainScript:
    """Test the check_version.py script end-to-end."""

    def _run(
        self, github_ref: str, github_output: Path | None = None, *, root: Path = ROOT
    ) -> subprocess.CompletedProcess:
        env = {"GITHUB_REF": github_ref, "PATH": ""}
        if github_output is not None:
            env["GITHUB_OUTPUT"] = str(github_output)

        return subprocess.run(
            [sys.executable, str(root / ".github" / "check_version.py")],
            capture_output=True,
            text=True,
            env=env,
        )

    def test_matching_tag(self):
        version = cargo_to_python_version(get_cargo_version())
        result = self._run(f"refs/tags/v{version}")
        assert result.returncode == 0
        assert "OK" in result.stdout

    @pytest.mark.parametrize(
        "cargo_version, tag_version, prerelease",
        [
            ("0.60.0", "0.60.0", "false"),
            ("0.61.0-alpha.1", "0.61.0a1", "true"),
            ("0.61.0-beta.2", "0.61.0b2", "true"),
            ("0.61.0-rc.1", "0.61.0-rc.1", "true"),
            ("0.60.0+build-1", "0.60.0+build-1", "false"),
        ],
    )
    def test_matching_tag_sets_github_output(
        self, tmp_path, cargo_version, tag_version, prerelease
    ):
        (tmp_path / ".github").mkdir()
        (tmp_path / ".github" / "check_version.py").write_text(
            (ROOT / ".github" / "check_version.py").read_text()
        )
        (tmp_path / "Cargo.toml").write_text(
            f'[package]\nversion = "{cargo_version}"\n'
        )
        github_output = tmp_path / "github-output"

        result = self._run(f"refs/tags/v{tag_version}", github_output, root=tmp_path)

        assert result.returncode == 0
        assert github_output.read_text() == f"is_prerelease={prerelease}\n"

    def test_mismatched_tag(self):
        result = self._run("refs/tags/v0.0.0")
        assert result.returncode == 1
        assert "MISMATCH" in result.stdout

    def test_not_a_tag(self):
        result = self._run("refs/heads/main")
        assert result.returncode == 1
        assert "Not a tag ref" in result.stdout


class TestDocsRelease:
    """Exercise the deployment guard with paginated API responses."""

    @pytest.mark.parametrize(
        "tag, releases, expected",
        [
            (
                "v0.61.10",
                [("v0.61.9", False, False), ("v0.61.10", False, False)],
                "true",
            ),
            (
                "v0.61.9",
                [("v0.61.9", False, False), ("v0.61.10", False, False)],
                "false",
            ),
            (
                "v0.60.99",
                [("v0.60.99", False, False), ("0.61.0", False, False)],
                "false",
            ),
            (
                "0.61.0",
                [
                    ("v1.0.0", True, False),
                    ("v0.62.0", False, True),
                    ("v0.62.0-rc.1", False, False),
                    ("unrelated-tag", False, False),
                    ("0.61.0", False, False),
                ],
                "true",
            ),
            ("v0.62.0", [("v0.61.0", False, False)], "false"),
            ("v0.61.0", [], None),
            ("v0.61.0", [("v0.61.0", False, True)], None),
        ],
    )
    def test_deployment_selection(self, tmp_path, tag, releases, expected):
        # One release per page also checks that a newer version on a later
        # API page wins over an older, more recently published maintenance tag.
        pages = [
            [{"tag_name": name, "draft": draft, "prerelease": prerelease}]
            for name, draft, prerelease in releases
        ]
        github_output = tmp_path / "github-output"
        result = subprocess.run(
            [sys.executable, str(ROOT / ".github" / "check_docs_release.py")],
            input=json.dumps(pages),
            capture_output=True,
            text=True,
            env={
                "GITHUB_REF": f"refs/tags/{tag}",
                "GITHUB_OUTPUT": str(github_output),
                "PATH": "",
            },
        )

        if expected is None:
            assert result.returncode == 1
            assert "No published stable version found" in result.stderr
            assert not github_output.exists()
        else:
            assert result.returncode == 0, result.stderr
            assert github_output.read_text() == f"publish={expected}\n"
