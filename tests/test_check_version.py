"""Tests for .github/check_version.py."""

import subprocess
import sys
from pathlib import Path

# Import the module from .github/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".github"))
from check_version import cargo_to_python_version, get_cargo_version  # noqa: E402


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

    def _run(self, github_ref: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(ROOT / ".github" / "check_version.py")],
            capture_output=True,
            text=True,
            env={"GITHUB_REF": github_ref, "PATH": ""},
        )

    def test_matching_tag(self):
        version = cargo_to_python_version(get_cargo_version())
        result = self._run(f"refs/tags/v{version}")
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_mismatched_tag(self):
        result = self._run("refs/tags/v0.0.0")
        assert result.returncode == 1
        assert "MISMATCH" in result.stdout

    def test_not_a_tag(self):
        result = self._run("refs/heads/main")
        assert result.returncode == 1
        assert "Not a tag ref" in result.stdout
