# PyFixest tests

Always run tests through pixi so Python dependencies, R packages, and the Rust
extension match CI.

```bash
# Fast Python suite: excludes extended, R, plots, and HAC tests.
pixi run test-py

# One focused file in the Python/R environment without the forced coverage report.
pixi run -e py312-r pytest tests/test_iv.py -x -q --no-cov

# External-reference suites.
pixi run test-r-core
pixi run test-r-extended
pixi run test-r-fixest
pixi run test-r-hac
```

The main cross-language regression matrix is
[`test_vs_fixest.py`](test_vs_fixest.py). Kernel-level tests such as
[`test_demean.py`](test_demean.py), [`test_crv1.py`](test_crv1.py), and
[`test_hac_meat.py`](test_hac_meat.py) compare optimized implementations with
readable references; model-family and post-estimation tests exercise public
entry points end to end.

R-dependent tests use `rpy2>=3.6` and the session converter in `conftest.py`;
there is no special pandas `<1.5.3` requirement. If a new test module imports
`rpy2`, add its filename to `_rpy2_test_files` in `conftest.py` so non-R
environments skip collection cleanly. Use the strict markers declared in
`pyproject.toml`: `against_r_core`, `against_r_extended`, `extended`, `plots`,
and `hac`.
