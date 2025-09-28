"""
R test runner that executes FEOLS tests and caches results.
Uses standalone R script instead of rpy2.
"""
import json
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Dict, Any

# Add tests directory to path for imports
tests_dir = Path(__file__).parent.parent
sys.path.insert(0, str(tests_dir))

from config.feols_tests import TestSingleFitFeols


class FeolsRTestRunner:
    """Runs R FEOLS tests and manages caching."""

    def __init__(self, cache_dir: str = "data/cached_results"):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_absolute():
            # Make relative to tests directory
            tests_dir = Path(__file__).parent.parent
            self.cache_dir = tests_dir / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Path to R script
        self.r_script_path = Path(__file__).parent / "run_feols_tests.R"

        # Check if R script exists
        if not self.r_script_path.exists():
            raise FileNotFoundError(f"R script not found: {self.r_script_path}")

    def _get_cache_path(self, test_case: TestSingleFitFeols) -> Path:
        """Get cache file path for a test case."""
        return self.cache_dir / f"{test_case.test_id}_{test_case.get_hash()}.json"

    def _prepare_test_params(self, test_case: TestSingleFitFeols) -> Dict[str, Any]:
        """Prepare test parameters for R script."""
        return {
            'test_id': test_case.test_id,
            'formula': test_case.formula,
            'data_params': test_case.get_data_params(),
            'estimation_params': test_case.get_estimation_params(),
            'hash': test_case.get_hash()
        }

    def _run_r_test(self, test_case: TestSingleFitFeols) -> Dict[str, Any]:
        """Run single R FEOLS test using standalone R script."""
        # Prepare test parameters
        test_params = self._prepare_test_params(test_case)

        # Create temporary files for input/output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
            json.dump(test_params, input_file, indent=2)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
            output_path = output_file.name

        try:
            # Run R script
            cmd = ['Rscript', str(self.r_script_path), input_path, output_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                return {
                    'test_id': test_case.test_id,
                    'error': f"R script failed: {result.stderr}",
                    'hash': test_case.get_hash(),
                    'success': False
                }

            # Read results
            with open(output_path, 'r') as f:
                results = json.load(f)

            return results

        except subprocess.TimeoutExpired:
            return {
                'test_id': test_case.test_id,
                'error': "R script timeout (>5 minutes)",
                'hash': test_case.get_hash(),
                'success': False
            }
        except Exception as e:
            return {
                'test_id': test_case.test_id,
                'error': f"Python error: {str(e)}",
                'hash': test_case.get_hash(),
                'success': False
            }
        finally:
            # Clean up temporary files
            try:
                Path(input_path).unlink()
                Path(output_path).unlink()
            except:
                pass

    def run_and_cache_test(self, test_case: TestSingleFitFeols, force_refresh: bool = False) -> Dict[str, Any]:
        """Run R test and cache results."""
        cache_path = self._get_cache_path(test_case)

        # Check if cached result exists and is valid
        if cache_path.exists() and not force_refresh:
            with open(cache_path, 'r') as f:
                cached_result = json.load(f)
                if cached_result.get('hash') == test_case.get_hash():
                    return cached_result

        # Run R test
        print(f"Running R test: {test_case.test_id}")
        results = self._run_r_test(test_case)

        # Cache results
        with open(cache_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_all_tests(self, test_cases: list, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Run all FEOLS tests and cache results."""
        results = {}

        for i, test_case in enumerate(test_cases):
            print(f"Progress: {i+1}/{len(test_cases)} - {test_case.test_id}")
            try:
                results[test_case.test_id] = self.run_and_cache_test(test_case, force_refresh)
            except Exception as e:
                print(f"Error running {test_case.test_id}: {e}")
                results[test_case.test_id] = {'error': str(e)}

        return results
