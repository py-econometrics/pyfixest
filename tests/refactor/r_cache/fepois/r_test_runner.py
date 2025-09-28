"""
FEPOIS-specific R test runner implementation.
"""
from pathlib import Path
from typing import Dict, Any
import sys

# Add tests directory to path for imports
tests_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(tests_dir))

from refactor.config.fepois.test_cases import TestSingleFitFepois
from refactor.r_cache.base.r_test_runner import BaseRTestRunner


class FepoisRTestRunner(BaseRTestRunner):
    """FEPOIS-specific R test runner implementation."""

    @property
    def test_method(self) -> str:
        """Return the test method name."""
        return "fepois"

    def _get_r_script_path(self) -> Path:
        """Return the path to the FEPOIS R script."""
        return Path(__file__).parent / "run_fepois_tests.R"
