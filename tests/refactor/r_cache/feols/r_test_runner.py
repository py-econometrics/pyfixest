"""
FEOLS-specific R test runner implementation.
"""
from pathlib import Path
from typing import Dict, Any
import sys

# Add tests directory to path for imports
tests_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(tests_dir))

from refactor.config.feols.test_cases import TestSingleFitFeols
from refactor.r_cache.base.r_test_runner import BaseRTestRunner


class FeolsRTestRunner(BaseRTestRunner):
    """FEOLS-specific R test runner implementation."""

    @property
    def test_method(self) -> str:
        """Return the test method name."""
        return "feols"

    def _get_r_script_path(self) -> Path:
        """Return the path to the FEOLS R script."""
        return Path(__file__).parent / "run_feols_tests.R"