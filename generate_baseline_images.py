#!/usr/bin/env python3
"""
Script to generate baseline images for decompose plot tests.
Run this script to create the baseline images that the tests will compare against.
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

from matplotlib import pyplot as plt

import pyfixest as pf
from pyfixest.utils.dgps import gelbach_data


def setup_baseline_dir():
    """Create the baseline images directory if it doesn't exist."""
    baseline_dir = Path("tests") / "baseline_images" / "test_decompose_plot"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    return baseline_dir


def generate_baseline_neg_y_plus_2x23(baseline_dir):
    """Generate baseline for y = -y + 2*x23 transformation."""
    print("Generating baseline for test_coefplot_neg_y_plus_2x23...")

    np.random.seed(12345)
    data = gelbach_data(nobs=1_000)
    data["y"] = -data["y"] + 2 * data["x23"]

    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
    gb = fit.decompose(
        decomp_var="x1",
        combine_covariates={"g1": ["x21"], "g2": ["x22", "x23"]},
        only_coef=True,
    )

    gb.coefplot(title="", figsize=(12, 8))

    output_path = baseline_dir / "test_coefplot_neg_y_plus_2x23.png"
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_baseline_neg_y_plus_1x23(baseline_dir):
    """Generate baseline for y = -y + 1*x23 transformation."""
    print("Generating baseline for test_coefplot_neg_y_plus_1x23...")

    np.random.seed(12345)
    data = gelbach_data(nobs=1_000)
    data["y"] = -data["y"] + 1 * data["x23"]

    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
    gb = fit.decompose(
        decomp_var="x1",
        combine_covariates={"g1": ["x21"], "g2": ["x22", "x23"]},
        only_coef=True,
    )

    gb.coefplot(title="", figsize=(12, 8))

    output_path = baseline_dir / "test_coefplot_neg_y_plus_1x23.png"
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_baseline_neg_y(baseline_dir):
    """Generate baseline for y = -y transformation."""
    print("Generating baseline for test_coefplot_neg_y...")

    np.random.seed(12345)
    data = gelbach_data(nobs=1_000)
    data["y"] = -data["y"]

    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
    gb = fit.decompose(
        decomp_var="x1",
        combine_covariates={"g1": ["x21"], "g2": ["x22", "x23"]},
        only_coef=True,
    )

    gb.coefplot(title="", figsize=(12, 8))

    output_path = baseline_dir / "test_coefplot_neg_y.png"
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Generate all baseline images."""
    print("Generating baseline images for decompose plot tests...")
    print("=" * 60)

    # Setup directory
    baseline_dir = setup_baseline_dir()
    print(f"Baseline directory: {baseline_dir}")
    print()

    # Generate each baseline
    generate_baseline_neg_y_plus_2x23(baseline_dir)
    generate_baseline_neg_y_plus_1x23(baseline_dir)
    generate_baseline_neg_y(baseline_dir)

    print()
    print("=" * 60)
    print("All baseline images generated successfully!")
    print(f"Images saved to: {baseline_dir}")
    print()
    print("You can now run the tests with:")
    print("  pytest tests/test_decompose_plot.py")


if __name__ == "__main__":
    main()
