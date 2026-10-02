"""Measure rank-check plus OLS fitting with and without factor reuse.

Run with ``pixi run python benchmarks/bench_ols_factor_reuse.py``. Uses a
single BLAS thread and seeded within-design arrays; formula preparation and
demeaning are deliberately outside this numerical-path benchmark.
"""

from __future__ import annotations

import argparse
import csv
import sys
from functools import partial
from statistics import median
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from pyfixest.estimation.internals.collinearity import drop_multicollinear_variables
from pyfixest.estimation.internals.fit_ import fit_ols


def _fit(reuse, *, X, Y, weights, names):
    design, _, factor = drop_multicollinear_variables(
        X=X, names=names, collin_tol=1e-9, weights=weights
    )
    return fit_ols(
        X=design,
        Y=Y,
        weights=weights,
        solver="scipy.linalg.solve",
        factorization=factor if reuse else None,
    )


def main() -> None:
    """Report alternating-order median timings as CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=9)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    writer = csv.writer(sys.stdout)
    writer.writerow(["n", "k", "weighted", "fresh_ms", "reuse_ms", "speedup"])
    with threadpool_limits(limits=1):
        for n, k in [(1000, 10), (10000, 100), (2000, 300)]:
            rng = np.random.default_rng(4204)
            X = rng.normal(size=(n, k))
            Y = X @ rng.normal(size=(k, 1)) + rng.normal(size=(n, 1))
            names = [f"x{i}" for i in range(k)]
            for weighted in [False, True]:
                weights = rng.uniform(0.1, 2.0, size=n) if weighted else None

                fit = partial(_fit, X=X, Y=Y, weights=weights, names=names)

                fresh = fit(False)
                reused = fit(True)
                # Well-conditioned seeded systems should agree near roundoff.
                np.testing.assert_allclose(
                    reused.beta,
                    fresh.beta,
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg="benchmark coefficients",
                )
                np.testing.assert_allclose(
                    reused.sandwich.bread,
                    fresh.sandwich.bread,
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg="benchmark bread",
                )
                timings = {False: [], True: []}
                for iteration in range(args.repeat):
                    # Alternate order to reduce systematic warm-cache bias.
                    for reuse in [False, True] if iteration % 2 == 0 else [True, False]:
                        start = perf_counter()
                        fit(reuse)
                        timings[reuse].append(perf_counter() - start)
                fresh_time = median(timings[False])
                reuse_time = median(timings[True])
                writer.writerow(
                    [
                        n,
                        k,
                        weighted,
                        f"{fresh_time * 1000:.3f}",
                        f"{reuse_time * 1000:.3f}",
                        f"{fresh_time / reuse_time:.3f}",
                    ]
                )


if __name__ == "__main__":
    main()
