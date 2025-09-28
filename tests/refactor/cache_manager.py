"""
Unified cache manager for all test methods.

This module provides a single interface for managing cached R test results
across all test methods (FEOLS, FEPOIS, IV, etc.).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod

# Add tests directory to path for imports
tests_dir = Path(__file__).parent.parent
sys.path.insert(0, str(tests_dir))

from refactor.config.base.test_cases import BaseTestCase
from refactor.r_cache.base.r_test_runner import BaseRTestRunner


class TestMethodRegistry:
    """Registry for available test methods and their implementations."""

    def __init__(self):
        self._methods: Dict[str, Dict[str, Any]] = {}
        self._register_available_methods()

    def _register_available_methods(self):
        """Register all available test methods."""
        # Register FEOLS (currently implemented)
        try:
            from refactor.config.feols.test_generator import generate_feols_test_cases
            from refactor.r_cache.feols.r_test_runner import FeolsRTestRunner

            self._methods["feols"] = {
                "generator": generate_feols_test_cases,
                "runner_class": FeolsRTestRunner,
                "description": "FEOLS (Fixed Effects OLS) tests",
                "original_function": "test_single_fit_feols",
                "status": "implemented"
            }
        except ImportError:
            pass

        # Register FEPOIS (implemented)
        try:
            from refactor.config.fepois.test_generator import generate_fepois_test_cases
            from refactor.r_cache.fepois.r_test_runner import FepoisRTestRunner

            self._methods["fepois"] = {
                "generator": generate_fepois_test_cases,
                "runner_class": FepoisRTestRunner,
                "description": "FEPOIS (Fixed Effects Poisson) tests",
                "original_function": "test_single_fit_fepois",
                "status": "implemented"
            }
        except ImportError:
            self._methods["fepois"] = {
                "generator": None,
                "runner_class": None,
                "description": "FEPOIS (Fixed Effects Poisson) tests",
                "original_function": "test_single_fit_fepois",
                "status": "planned"
            }

        # Register IV (planned)
        self._methods["iv"] = {
            "generator": None,
            "runner_class": None,
            "description": "IV (Instrumental Variables) tests",
            "original_function": "test_single_fit_iv",
            "status": "planned"
        }

    def get_available_methods(self) -> List[str]:
        """Get list of all available test methods."""
        return list(self._methods.keys())

    def get_implemented_methods(self) -> List[str]:
        """Get list of implemented test methods."""
        return [method for method, info in self._methods.items()
                if info["status"] == "implemented"]

    def get_method_info(self, method: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific test method."""
        return self._methods.get(method)

    def is_implemented(self, method: str) -> bool:
        """Check if a test method is implemented."""
        info = self.get_method_info(method)
        return info is not None and info["status"] == "implemented"


class UnifiedCacheManager:
    """
    Unified cache manager for all test methods.

    Provides a single interface for generating, clearing, and managing
    cached R test results across all test methods.
    """

    def __init__(self, cache_base_dir: str = "data/cached_results"):
        """
        Initialize the unified cache manager.

        Parameters:
        -----------
        cache_base_dir : str
            Base directory for caching results, relative to refactor directory
        """
        self.cache_base_dir = Path(cache_base_dir)
        if not self.cache_base_dir.is_absolute():
            # Make relative to refactor directory
            refactor_dir = Path(__file__).parent
            self.cache_base_dir = refactor_dir / cache_base_dir
        self.cache_base_dir.mkdir(parents=True, exist_ok=True)

        self.registry = TestMethodRegistry()

    def generate_cache(self, methods: Optional[List[str]] = None,
                      force_refresh: bool = False, n_jobs: int = -1) -> Dict[str, Dict[str, Any]]:
        """
        Generate cached results for specified test methods.

        Parameters:
        -----------
        methods : Optional[List[str]]
            List of test methods to generate cache for. If None, generates for all implemented methods.
        force_refresh : bool
            Whether to force refresh of existing cache
        n_jobs : int
            Number of parallel jobs. -1 means use all available cores, 1 means sequential

        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Results summary for each method
        """
        if methods is None:
            methods = self.registry.get_implemented_methods()

        results_summary = {}

        for method in methods:
            if not self.registry.is_implemented(method):
                print(f"⚠️  Method '{method}' is not implemented yet. Skipping.")
                results_summary[method] = {"status": "not_implemented"}
                continue

            print(f"\n🔄 Generating cache for {method.upper()}...")

            method_info = self.registry.get_method_info(method)
            generator = method_info["generator"]
            runner_class = method_info["runner_class"]

            # Generate test cases
            test_cases = generator()
            print(f"Generated {len(test_cases)} {method.upper()} test cases")

            # Run tests and cache results
            runner = runner_class(str(self.cache_base_dir))
            results = runner.run_all_tests(test_cases, force_refresh, n_jobs=n_jobs)

            # Create summary
            successful = len([r for r in results.values() if r.get('success', True) and 'error' not in r])
            failed = len([r for r in results.values() if 'error' in r or not r.get('success', True)])

            summary = {
                'total_tests': len(results),
                'successful_tests': successful,
                'failed_tests': failed,
                'test_method': method,
                'last_updated': datetime.now().isoformat(),
                'status': 'completed'
            }

            # Save method-specific summary
            method_cache_dir = self.cache_base_dir / method
            method_cache_dir.mkdir(exist_ok=True)
            summary_path = method_cache_dir / f"{method}_cache_summary.json"

            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            results_summary[method] = summary

            print(f"✅ {method.upper()}: Generated {successful} cached results")
            if failed > 0:
                print(f"❌ {method.upper()}: Failed to generate {failed} results")

        return results_summary

    def clear_cache(self, methods: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Clear cached results for specified test methods.

        Parameters:
        -----------
        methods : Optional[List[str]]
            List of test methods to clear cache for. If None, clears all methods.

        Returns:
        --------
        Dict[str, int]
            Number of files cleared for each method
        """
        if methods is None:
            methods = self.registry.get_available_methods()

        cleared_summary = {}

        for method in methods:
            method_cache_dir = self.cache_base_dir / method
            cleared = 0

            if method_cache_dir.exists():
                # Clear test result files
                for cache_file in method_cache_dir.glob(f"{method}_*.json"):
                    cache_file.unlink()
                    cleared += 1

                # Clear summary file
                summary_path = method_cache_dir / f"{method}_cache_summary.json"
                if summary_path.exists():
                    summary_path.unlink()
                    cleared += 1

            cleared_summary[method] = cleared
            print(f"🗑️  Cleared {cleared} {method.upper()} cache files")

        return cleared_summary

    def show_summary(self, methods: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Show cache summary for specified test methods.

        Parameters:
        -----------
        methods : Optional[List[str]]
            List of test methods to show summary for. If None, shows all methods.

        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Summary information for each method
        """
        if methods is None:
            methods = self.registry.get_available_methods()

        summaries = {}

        print("📊 Cache Summary:")
        print("=" * 50)

        for method in methods:
            method_cache_dir = self.cache_base_dir / method
            summary_path = method_cache_dir / f"{method}_cache_summary.json"

            if not summary_path.exists():
                print(f"\n{method.upper()}:")
                print(f"  Status: No cache found")
                if self.registry.is_implemented(method):
                    print(f"  Action: Run 'generate {method}' to create cache")
                else:
                    print(f"  Action: Method not implemented yet")
                summaries[method] = {"status": "no_cache"}
                continue

            with open(summary_path, 'r') as f:
                summary = json.load(f)

            print(f"\n{method.upper()}:")
            print(f"  Total tests: {summary['total_tests']}")
            print(f"  Successful: {summary['successful_tests']}")
            print(f"  Failed: {summary['failed_tests']}")
            print(f"  Last updated: {summary.get('last_updated', 'unknown')}")

            # Show cache directory info
            if method_cache_dir.exists():
                cache_files = list(method_cache_dir.glob(f"{method}_*.json"))
                cache_files = [f for f in cache_files if not f.name.endswith('_summary.json')]
                print(f"  Cache files: {len(cache_files)}")

                if cache_files:
                    total_size = sum(f.stat().st_size for f in cache_files)
                    print(f"  Total size: {total_size / (1024*1024):.1f} MB")

            summaries[method] = summary

        return summaries

    def list_methods(self):
        """List all available test methods and their status."""
        print("🔍 Available Test Methods:")
        print("=" * 50)

        for method in self.registry.get_available_methods():
            info = self.registry.get_method_info(method)
            status_emoji = "✅" if info["status"] == "implemented" else "⏳"

            print(f"\n{status_emoji} {method.upper()}")
            print(f"  Description: {info['description']}")
            print(f"  Original function: {info['original_function']}")
            print(f"  Status: {info['status']}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Unified cache manager for all test methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate                    # Generate cache for all implemented methods
  %(prog)s generate feols              # Generate cache for FEOLS only
  %(prog)s generate feols fepois       # Generate cache for FEOLS and FEPOIS
  %(prog)s clear                       # Clear cache for all methods
  %(prog)s clear feols                 # Clear cache for FEOLS only
  %(prog)s summary                     # Show summary for all methods
  %(prog)s list                        # List all available methods
        """
    )

    parser.add_argument("command",
                       choices=["generate", "clear", "summary", "list"],
                       help="Command to run")
    parser.add_argument("methods", nargs="*",
                       help="Test methods to operate on (e.g., feols, fepois, iv). If not specified, operates on all available methods.")
    parser.add_argument("--force", action="store_true",
                       help="Force refresh of existing cache")
    parser.add_argument("--n-jobs", type=int, default=-1,
                       help="Number of parallel jobs for R test execution. -1 uses all cores, 1 is sequential (default: -1)")

    args = parser.parse_args()

    manager = UnifiedCacheManager()

    methods = args.methods if args.methods else None

    if args.command == "generate":
        manager.generate_cache(methods, args.force, n_jobs=args.n_jobs)
    elif args.command == "clear":
        manager.clear_cache(methods)
    elif args.command == "summary":
        manager.show_summary(methods)
    elif args.command == "list":
        manager.list_methods()


if __name__ == "__main__":
    main()
