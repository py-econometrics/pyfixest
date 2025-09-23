#!/usr/bin/env python3
"""
Test the new cached testing system.

This script validates that the configuration, R generation, and Python tests
all work together correctly.
"""

import sys
from pathlib import Path
import time

# Add config directory to path
config_dir = Path(__file__).parent.parent / "config"
sys.path.insert(0, str(config_dir))

from config_loader import TestConfigLoader, CachedResultsLoader
from run_r_generation import check_r_available, check_results_generated


def test_config_loader():
    """Test that configuration loading works."""
    print("Testing configuration loader...")

    try:
        config = TestConfigLoader()

        # Test basic functionality
        feols_formulas = config.get_formulas("feols")
        assert len(feols_formulas) > 0, "No FEOLS formulas found"

        feols_config = config.get_test_config("feols")
        assert "inference_types" in feols_config, "Missing inference_types in FEOLS config"

        tolerance = config.get_tolerance("default")
        assert "rtol" in tolerance, "Missing rtol in tolerance"

        # Test combination generation
        combinations = config.generate_test_combinations("feols")
        assert len(combinations) > 0, "No test combinations generated"

        print(f"✓ Config loader works. Found {len(feols_formulas)} FEOLS formulas, {len(combinations)} test combinations")
        return True

    except Exception as e:
        print(f"✗ Config loader failed: {e}")
        return False


def test_cache_loader():
    """Test that cache loading works (if cache exists)."""
    print("Testing cache loader...")

    try:
        cache = CachedResultsLoader()

        if cache.is_cache_valid():
            available_types = cache.list_available_test_types()
            print(f"✓ Cache is valid. Available types: {available_types}")

            # Try loading one result
            if "feols" in available_types:
                feols_results = cache.load_results("feols")
                print(f"✓ Loaded {len(feols_results)} FEOLS cached results")

            return True
        else:
            print("! Cache not found (this is OK if R results haven't been generated yet)")
            return True  # Not an error

    except Exception as e:
        print(f"✗ Cache loader failed: {e}")
        return False


def test_r_environment():
    """Test R environment setup."""
    print("Testing R environment...")

    r_ok, r_msg = check_r_available()
    if r_ok:
        print(f"✓ {r_msg}")
        return True
    else:
        print(f"✗ R environment issue: {r_msg}")
        return False


def run_mini_python_test():
    """Run a minimal Python test to verify the system works."""
    print("Testing Python test execution...")

    try:
        # Import pyfixest
        import pyfixest as pf

        # Load config
        config = TestConfigLoader()

        # Get simple test data
        params = config.get_data_params("feols")
        data = pf.get_data(N=100, seed=123)  # Smaller dataset for quick test

        # Run a simple model
        mod = pf.feols("Y ~ X1", data=data)
        coef = mod.coef().xs("X1")

        assert not pd.isna(coef), "Failed to get coefficient"

        print(f"✓ Basic Python test passed. X1 coefficient: {coef:.4f}")
        return True

    except Exception as e:
        print(f"✗ Python test failed: {e}")
        return False


def main():
    """Run all system tests."""
    print("=" * 50)
    print("Testing Cached PyFixest System")
    print("=" * 50)

    tests = [
        ("Configuration Loading", test_config_loader),
        ("Cache Loading", test_cache_loader),
        ("R Environment", test_r_environment),
        ("Python Testing", run_mini_python_test)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))

    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)

    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False

    print()
    if all_passed:
        print("✓ All system tests passed!")

        # Check if cache exists
        results_exist, results_msg = check_results_generated()
        if results_exist:
            print("✓ Cached results found - system ready for fast testing")
        else:
            print("! Cached results not found")
            print("  Run 'python tests/scripts/run_r_generation.py' to generate them")

        return 0
    else:
        print("✗ Some tests failed - check setup")
        return 1


if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues if not available
    sys.exit(main())
