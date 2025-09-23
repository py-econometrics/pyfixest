#!/usr/bin/env python3
"""
Simple script to check cached R results status.
"""

import sys
from pathlib import Path

import pandas as pd


def check_cache_status():
    """Check if cached results exist and provide status information."""
    test_dir = Path(__file__).parent.parent
    cache_dir = test_dir / "data" / "cached_results"

    if not cache_dir.exists():
        print("❌ Cache directory does not exist")
        print(f"   Expected: {cache_dir}")
        return False

    expected_files = [
        "feols_results.csv",
        "iv_results.csv",
        "glm_results.csv",
        "fepois_results.csv",
        "metadata.csv",
    ]

    existing_files = []
    missing_files = []

    for filename in expected_files:
        filepath = cache_dir / filename
        if filepath.exists():
            existing_files.append(filename)
        else:
            missing_files.append(filename)

    print(f"📁 Cache directory: {cache_dir}")
    print(f"✅ Existing files: {len(existing_files)}/{len(expected_files)}")

    if existing_files:
        print("   Found:")
        for filename in existing_files:
            filepath = cache_dir / filename
            try:
                if filename != "metadata.csv":
                    df = pd.read_csv(filepath)
                    print(f"   - {filename}: {len(df)} rows")
                else:
                    df = pd.read_csv(filepath)
                    print(
                        f"   - {filename}: generated at {df['generated_at'].iloc[0] if 'generated_at' in df.columns else 'unknown'}"
                    )
            except Exception as e:
                print(f"   - {filename}: ⚠️  Error reading file ({e})")

    if missing_files:
        print(f"❌ Missing files: {len(missing_files)}")
        for filename in missing_files:
            print(f"   - {filename}")
        print("\n💡 To generate missing files, run:")
        print("   pixi run --environment r-test-gen generate-r-results")
        return False

    print("\n✅ All cached results are present!")
    return True


def main():
    """Main function."""
    print("🔍 Checking cached R results status...\n")

    cache_ok = check_cache_status()

    if cache_ok:
        print("\n🚀 Ready to run fast cached tests:")
        print("   pixi run --environment dev tests-cached")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
