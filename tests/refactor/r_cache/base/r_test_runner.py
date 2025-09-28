"""
Base R test runner with shared functionality.

This module provides the abstract base class for R test runners that handle
execution of R scripts and caching of results.
"""
import json
import subprocess
import tempfile
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List
from joblib import Parallel, delayed

# Add tests directory to path for imports
tests_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(tests_dir))

from refactor.config.base.test_cases import BaseTestCase


def run_single_test_worker(test_case: BaseTestCase, cache_dir: str, runner_class, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Generic worker function to run a single R test.
    This function is designed to be used with joblib.delayed for parallel execution.

    Parameters:
    -----------
    test_case : BaseTestCase
        The test case to run
    cache_dir : str
        Directory for caching results
    runner_class : class
        The specific runner class to instantiate
    force_refresh : bool
        Whether to force refresh of existing cache
    """
    try:
        # Create a temporary runner instance for this worker
        runner = runner_class(cache_dir)
        result = runner.run_and_cache_test(test_case, force_refresh)
        print(f"✓ Completed {test_case.test_id}")
        return result
    except Exception as e:
        print(f"✗ Failed {test_case.test_id}: {e}")
        return {
            'test_id': test_case.test_id,
            'error': str(e),
            'hash': test_case.get_hash(),
            'success': False
        }


class BaseRTestRunner(ABC):
    """
    Abstract base class for R test runners.

    Provides common functionality for executing R scripts, managing cache,
    and handling parallel execution.
    """

    def __init__(self, cache_dir: str = "data/cached_results"):
        """
        Initialize the R test runner.

        Parameters:
        -----------
        cache_dir : str
            Directory for caching results, relative to refactor directory
        """
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_absolute():
            # Make relative to refactor directory
            refactor_dir = Path(__file__).parent.parent.parent
            self.cache_dir = refactor_dir / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Path to method-specific R script (to be set by subclasses)
        self.r_script_path = None

    @property
    @abstractmethod
    def test_method(self) -> str:
        """Return the test method name (e.g., 'feols', 'fepois', 'iv')."""
        pass

    @abstractmethod
    def _get_r_script_path(self) -> Path:
        """Return the path to the R script for this test method."""
        pass

    def _get_cache_path(self, test_case: BaseTestCase) -> Path:
        """Generate cache file path for a test case."""
        method_dir = self.cache_dir / test_case.test_method
        method_dir.mkdir(exist_ok=True)
        filename = f"{test_case.test_id}_{test_case.get_hash()}.json"
        return method_dir / filename

    def _run_r_test(self, test_case: BaseTestCase) -> Dict[str, Any]:
        """
        Execute R test for a single test case.

        This method handles the communication with R via temporary files
        and returns the parsed results.
        """
        if self.r_script_path is None:
            self.r_script_path = self._get_r_script_path()

        if not self.r_script_path.exists():
            raise FileNotFoundError(f"R script not found: {self.r_script_path}")

        # Prepare test parameters
        test_params = test_case.to_dict()
        test_params['hash'] = test_case.get_hash()

        # Create temporary files for R communication
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
            json.dump(test_params, input_file, indent=2)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
            output_path = output_file.name

        try:
            # Execute R script
            cmd = ['Rscript', str(self.r_script_path), input_path, output_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

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

    def run_and_cache_test(self, test_case: BaseTestCase, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run a single test and cache the results.

        Parameters:
        -----------
        test_case : BaseTestCase
            The test case to run
        force_refresh : bool
            Whether to force refresh of existing cache

        Returns:
        --------
        Dict[str, Any]
            Test results dictionary
        """
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

    def run_all_tests(self, test_cases: List[BaseTestCase], force_refresh: bool = False, n_jobs: int = -1) -> Dict[str, Dict[str, Any]]:
        """
        Run all tests and cache results.

        Parameters:
        -----------
        test_cases : List[BaseTestCase]
            List of test cases to run
        force_refresh : bool
            Whether to force refresh of existing cache
        n_jobs : int
            Number of parallel jobs. -1 means use all available cores, 1 means sequential

        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Dictionary mapping test_id to results
        """
        if n_jobs == 1:
            # Sequential execution
            return self._run_all_tests_sequential(test_cases, force_refresh)
        else:
            # Parallel execution with joblib
            return self._run_all_tests_parallel(test_cases, force_refresh, n_jobs)

    def _run_all_tests_sequential(self, test_cases: List[BaseTestCase], force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Run all tests sequentially."""
        results = {}
        total = len(test_cases)

        print(f"Running {total} R tests sequentially...")
        for i, test_case in enumerate(test_cases):
            print(f"Progress: {i+1}/{total} - {test_case.test_id}")
            try:
                results[test_case.test_id] = self.run_and_cache_test(test_case, force_refresh)
            except Exception as e:
                print(f"Error running {test_case.test_id}: {e}")
                results[test_case.test_id] = {'error': str(e)}

        return results

    def _run_all_tests_parallel(self, test_cases: List[BaseTestCase], force_refresh: bool = False, n_jobs: int = -1) -> Dict[str, Dict[str, Any]]:
        """Run all tests in parallel using joblib."""
        total = len(test_cases)

        # Determine actual number of jobs
        if n_jobs == -1:
            import multiprocessing
            actual_jobs = multiprocessing.cpu_count()
        else:
            actual_jobs = min(n_jobs, total)

        print(f"Running {total} R tests in parallel with {actual_jobs} workers using joblib...")

        # Use joblib to run tests in parallel
        results_list = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(run_single_test_worker)(test_case, str(self.cache_dir), self.__class__, force_refresh)
            for test_case in test_cases
        )

        # Convert list of results to dictionary
        results = {}
        for result in results_list:
            test_id = result.get('test_id', 'unknown')
            results[test_id] = result

        return results
