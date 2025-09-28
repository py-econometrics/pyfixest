"""
Script to manage FEOLS R test cache.
"""
import argparse
from pathlib import Path
import json
from datetime import datetime
import sys

# Add tests directory to path for imports
tests_dir = Path(__file__).parent.parent
sys.path.insert(0, str(tests_dir))

from refactor.config.feols_test_generator import generate_feols_test_cases
from refactor.r_cache.r_test_runner import FeolsRTestRunner


def generate_cache(force_refresh: bool = False):
    """Generate all R results and cache them."""
    runner = FeolsRTestRunner()
    test_cases = generate_feols_test_cases()

    print(f"Generating R cache for {len(test_cases)} FEOLS test cases...")
    results = runner.run_all_tests(test_cases, force_refresh)

    # Save summary
    summary_path = Path("data/cached_results/feols_cache_summary.json")
    summary = {
        'total_tests': len(results),
        'successful_tests': len([r for r in results.values() if 'error' not in r]),
        'failed_tests': len([r for r in results.values() if 'error' in r]),
        'test_group': 'feols',
        'last_updated': datetime.now().isoformat()
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Generated {summary['successful_tests']} cached results")
    if summary['failed_tests'] > 0:
        print(f"Failed to generate {summary['failed_tests']} results")

        # Show some failed tests
        failed_tests = [test_id for test_id, result in results.items() if 'error' in result]
        print("First 5 failed tests:")
        for test_id in failed_tests[:5]:
            print(f"  {test_id}: {results[test_id]['error']}")

    return results


def clear_cache():
    """Clear FEOLS cached results."""
    cache_dir = Path("data/cached_results")

    # Clear FEOLS cache files
    cleared = 0
    for cache_file in cache_dir.glob("feols_*.json"):
        cache_file.unlink()
        cleared += 1

    # Clear summary
    summary_path = cache_dir / "feols_cache_summary.json"
    if summary_path.exists():
        summary_path.unlink()
        cleared += 1

    print(f"Cleared {cleared} FEOLS cache files")


def show_summary():
    """Show cache summary."""
    summary_path = Path("data/cached_results/feols_cache_summary.json")

    if not summary_path.exists():
        print("No cache summary found. Run 'generate' first.")
        return

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    print("FEOLS Cache Summary:")
    print(f"  Total tests: {summary['total_tests']}")
    print(f"  Successful: {summary['successful_tests']}")
    print(f"  Failed: {summary['failed_tests']}")
    print(f"  Last updated: {summary.get('last_updated', 'unknown')}")

    # Show cache directory size
    cache_dir = Path("data/cached_results")
    cache_files = list(cache_dir.glob("feols_*.json"))
    print(f"  Cache files: {len(cache_files)}")

    if cache_files:
        total_size = sum(f.stat().st_size for f in cache_files)
        print(f"  Total size: {total_size / (1024*1024):.1f} MB")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Manage FEOLS R test cache")
    parser.add_argument("command", choices=["generate", "clear", "summary"],
                       help="Command to run")
    parser.add_argument("--force", action="store_true",
                       help="Force refresh of existing cache")

    args = parser.parse_args()

    if args.command == "generate":
        generate_cache(args.force)
    elif args.command == "clear":
        clear_cache()
    elif args.command == "summary":
        show_summary()


if __name__ == "__main__":
    main()
