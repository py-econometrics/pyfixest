#!/usr/bin/env python3
"""
Simple test to validate basic system components without pandas dependencies.
"""

import json
import sys
from pathlib import Path


def test_config_file():
    """Test that configuration file exists and is valid JSON."""
    print("Testing configuration file...")

    config_path = Path(__file__).parent.parent / "config" / "test_specifications.json"

    if not config_path.exists():
        print(f"✗ Configuration file not found: {config_path}")
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)

        # Basic validation
        required_keys = ["metadata", "data_generation", "test_configurations"]
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing required key: {key}")
                return False

        # Check test configurations
        test_types = ["feols", "iv", "glm", "fepois"]
        for test_type in test_types:
            if test_type not in config["test_configurations"]:
                print(f"✗ Missing test configuration: {test_type}")
                return False

            test_config = config["test_configurations"][test_type]
            if "formulas" not in test_config:
                print(f"✗ Missing formulas in {test_type}")
                return False

        print(
            f"✓ Configuration file valid with {len(config['test_configurations'])} test types"
        )
        return True

    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in configuration file: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading configuration: {e}")
        return False


def test_pyfixest_import():
    """Test that pyfixest can be imported and basic functionality works."""
    print("Testing pyfixest import...")

    try:
        import pyfixest as pf

        print("✓ pyfixest imported successfully")

        # Test basic data generation
        data = pf.get_data(N=50, seed=123)
        print(f"✓ Generated test data with {len(data)} rows")

        # Test basic model fitting
        mod = pf.feols("Y ~ X1", data=data)
        coef = mod.coef()
        print(f"✓ Basic model fitted, X1 coefficient: {coef.iloc[0]:.4f}")

        return True

    except ImportError as e:
        print(f"✗ Failed to import pyfixest: {e}")
        return False
    except Exception as e:
        print(f"✗ Error testing pyfixest: {e}")
        return False


def test_r_script_exists():
    """Test that R generation script exists."""
    print("Testing R script...")

    r_script = Path(__file__).parent / "generate_r_results.R"

    if not r_script.exists():
        print(f"✗ R script not found: {r_script}")
        return False

    # Basic validation - check it's an R script
    try:
        with open(r_script) as f:
            content = f.read()

        if "library(fixest)" not in content:
            print("✗ R script doesn't seem to load fixest")
            return False

        if "generate_all_results" not in content:
            print("✗ R script missing main function")
            return False

        print("✓ R script exists and looks valid")
        return True

    except Exception as e:
        print(f"✗ Error reading R script: {e}")
        return False


def test_directory_structure():
    """Test that required directories exist."""
    print("Testing directory structure...")

    base_dir = Path(__file__).parent.parent
    required_dirs = ["config", "scripts", "data", "data/cached_results"]

    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)

    if missing_dirs:
        print(f"✗ Missing directories: {missing_dirs}")
        return False

    print("✓ All required directories exist")
    return True


def main():
    """Run simple validation tests."""
    print("=" * 50)
    print("Simple System Validation")
    print("=" * 50)

    tests = [
        ("Directory Structure", test_directory_structure),
        ("Configuration File", test_config_file),
        ("R Script", test_r_script_exists),
        ("PyFixest Import", test_pyfixest_import),
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
        print("✓ Basic system validation passed!")
        print("\nNext steps:")
        print("1. Generate R results: python scripts/run_r_generation.py")
        print("2. Run cached tests: pytest test_vs_fixest_cached.py")
        return 0
    else:
        print("✗ Some basic tests failed - check setup")
        return 1


if __name__ == "__main__":
    sys.exit(main())
