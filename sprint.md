# PyFixest Hackathon


## Hackathon Workstreams

### WS 1: Demeaning Performance ([#1179](https://github.com/py-econometrics/pyfixest/issues/1179))

- Owner: Kristof
- Plan: Improve and benchmark the performance of the demeaner backend, building on Kristof's work.
- Done when: Python bindings of new algo published to PyPi, integrated into PyFixest as backend.
- Needs from others: none or 1 short line

- Create Rust crate `within` with fast demeaning backend.
- Provide Python and R bindings.
- Prepare and run extensive benchmarks of the new algorithm. See e.g. here: https://github.com/s3alfisc/fixest_benchmarks
- Prepare documentation that explains the different algorithmic solutions and help users decide when to use which solver.
- Provide a heuristic that chooses between new algo and default MAP for demeaning.
- Decide if we should keep all currently existing benchmarks: `numba`, `rust`, `scipy`, `cupy`, `jax`.
- GPU acceleration: The `scipy` and `cupy` backends run the demeaning via the LSMR algorithm. Most pyfixest users are on Mac. Should we add a torch implementation?
- In addition, the LSMR comes with different convergence checks than the MAP algo in `numba`, `rust`, `jax`. Does it make sense to hand-roll LSMR with a convergence
  check that matches the MAP algos?
- Varying slopes: Extend demeaning to handle varying-slopes specifications.
- **Convergence diagnostics** ([#1089](https://github.com/py-econometrics/pyfixest/issues/1089)): Report the number of iterations and final demeaning tolerance after convergence.
- **Warm starts** ([#1202](https://github.com/py-econometrics/pyfixest/issues/1202)): For multiple estimation, pyfixest "caches" results from prior runs. If the data structure is
  identical, the cached / already demeaned data is reused. If the data structure between fits is not identical, we run the demeaning again with "cold start". Using the demeaned
  results from the previous iteration should be more efficient (provided the data has a large overlap).
- Python

**Key files**: `src/demean.rs`, `pyfixest/core/demean.py`, `pyfixest/estimation/internals/demean_.py`, `pyfixest/estimation/numba/`, `pyfixest/estimation/jax/`, `pyfixest/estimation/cupy/`

---

### WS 2: Numba Deprecation

Goal: replace all `@nb.njit` functions with Rust (PyO3) and drop numba as a dependency.

**Already in Rust** (`src/` → `pyfixest/core/`): demeaning, CRV1, collinearity detection, nested FE counting. Wired up as the `"rust"` backend in `backends.py`.

**Still numba-only — needs Rust port:**

| Component | File | # njit funcs |
|-----------|------|:---:|
| Singleton detection | `internals/detect_singletons_.py` | 3 |
| HAC standard errors | `internals/vcov_utils.py` | 7 |
| Randomization inference | `post_estimation/ritest.py` | 4 |
| Quantile regression | `quantreg/quantreg_.py` | 1 |

**Migration steps:**

1. Port the above functions to `src/` and expose via `pyfixest/core/`
2. Update `backends.py` — JAX/CuPy fallbacks should use Rust instead of numba
3. Switch default `demeaner_backend` from `"numba"` to `"rust"` across all APIs
4. Delete `estimation/numba/`, remove all `import numba` / `@nb.njit`
5. Drop `numba` from `pyproject.toml` and `pixi.toml`
6. Update tests (backend parametrization) and docs

---

### WS 2: Scikit-learn Style API & Inheritance Refactor ([#1180](https://github.com/py-econometrics/pyfixest/issues/1180))

Expose `Feols`, `Fepois`, `Feglm`, `Feiv` as public classes with a clean API.

- Owner: tbd
- Plan: Significant cleanup of the PyFixest Code Base + statsmodels / scikit - style API
- Done when: `Feols.from_formula().fit()` syntax is tested and documented.

**Candidate backlog items:**

- Inheritance structure: Discuss and decide if we should introduce a `BaseRegression` class that all model classes inherit from (instead of everything inheriting from `Feols`).
- Support two calling conventions: `Feols().from_formula(fml, data).fit()` and `Feols().fit(X=X, y=y, fe=fe)`, similar to `statsmodels`. The "class first" APIs should
  coexist with the functional APIs.
- One complication is that several post-processing methods require input from the formula API; for example, the `Feols.summary` method prints a formula.
- Redistribute orchestration logic currently in `FixestMulti` into individual model classes where appropriate.
- Add tests and user-facing documentation.

**Key files**: `pyfixest/estimation/models/`, `pyfixest/estimation/FixestMulti_.py`

---

### WS 3: Formula Parsing & Narwhals ([#1182](https://github.com/py-econometrics/pyfixest/issues/1182))

- Owner: Leo
- Plan: Fully port internal data frame manipulation to `narwhals`
- Done when: `polars` data.frames are treated as first class citizens in pf's code base. `formulaic` operators are either accepted or rejected.

**Candidate backlog items:**

- Implement varying slopes syntax parsing and model matrix encoding.
- Full `narwhals` support: accept Polars DataFrames end-to-end without internal pandas conversion, return native Polars when input is Polars.
- Move the `^` operator for interacting fixed effects to `formulaic` parsing, if possible.ßßß
- Decide on adoption of formulaic `i` operator (currently in dev).
- Decide on adoption of formulaic IV syntax.

**Key files**: `pyfixest/estimation/formula/parse.py`, `pyfixest/estimation/formula/model_matrix.py`

---

### WS 4: Documentation Rewrite ([#1174](https://github.com/py-econometrics/pyfixest/issues/1174))

Reorganize and expand docs following the Diataxis framework.

- Owner: Alex
- Plan: Reorganize and expand docs following the Diataxis framework.
- Done when: docs look excellent.

**Candidate backlog items:**

- Revise/write tutorials: Fixed Effects, IV, GLM, Inference, Getting Started, Multiple Estimation.
- Add how-to guides: marginal effects, GPU usage, surrogate indices, experimental debiasing.
- EU Pay Gap Directive notebook (wage gap composition via Gelbach decomposition).
- Statistical documentation covering core formulas and methods.
- Textbook replication examples.

**Key files**: `docs/tutorials/`, `docs/how-to/`, `docs/explanation/`, `docs/_quarto.yml`, `docs/_sidebar.yml`

---

### WS 5: Smaller Standalone Tickets ([#1183](https://github.com/py-econometrics/pyfixest/issues/1183))

Independent issues good for shorter contributions. Pick any:

- `metrics`
- `rust`
- `swe`
- `good-first-issue`

#### WS 5 Intake Board

| Ticket/Idea | Description | Type | Owner |
|-------------|-------------|------|-------|
| [#1025](https://github.com/py-econometrics/pyfixest/issues/1025) | Automatic F-statistic in `feols()` | `good-first-issue` | `TBD` |
| [#859](https://github.com/py-econometrics/pyfixest/issues/859) | Code reorganization in large scripts | `good-first-issue` | `TBD` |
| [#1119](https://github.com/py-econometrics/pyfixest/issues/1119) | Benchmarking vs. linearmodels & statsmodels (see [here](https://github.com/s3alfisc/fixest_benchmarks))| `good-first-issue` | `TBD` |
| — | Make WLS weight application explicit (track raw vs. weighted arrays) | `good-first-issue` | `TBD` |
| [#1177](https://github.com/py-econometrics/pyfixest/issues/1177) | Bug: `fixef()` crashes on IV models | `good-first-issue` | `TBD` |
| [#559](https://github.com/py-econometrics/pyfixest/issues/559) | Show first-stage F-stat / MOP effective F in IV `summary()`/`etable()` | `good-first-issue` | `TBD` |
| — | GLM residual types: `"deviance"` / `"pearson"` / `"working"` in `resid()` | `good-first-issue` | `TBD` |
| [#1104](https://github.com/py-econometrics/pyfixest/issues/1104) | Migrate type checking from mypy to ty | `swe` | `TBD` |
| []() | pre-commit to prek | `swe` | `TBD` |
| [#1114](https://github.com/py-econometrics/pyfixest/issues/1114) | Snapshot tests for visualization functions | `swe` | `TBD` |
| [#1131](https://github.com/py-econometrics/pyfixest/issues/1131) | Python 3.14 / scipy >1.16 compatibility | `swe` | `TBD` |
| [#549](https://github.com/py-econometrics/pyfixest/issues/549) | Publish conda package | Issue | `TBD` |
| []() | Memory Manage Scan | `swe` | `TBD` |
| [#1203](https://github.com/py-econometrics/pyfixest/issues/1203) | Gelbach decomposition: analytical standard errors | `metrics` | `TBD` |
| [#1204](https://github.com/py-econometrics/pyfixest/issues/1204) | Gelbach decomposition: frequency & analytical weights | `metrics` | `TBD` |
| [#500](https://github.com/py-econometrics/pyfixest/issues/500) | Conley spatial standard errors | `metrics`, `rust` | `TBD` |
| [#428](https://github.com/py-econometrics/pyfixest/issues/428) | SUR variance-covariance matrix | `metrics` | `TBD` |
| [#695](https://github.com/py-econometrics/pyfixest/issues/695) | Offset support in Poisson models | `metrics` | `TBD` |
| [#1148](https://github.com/py-econometrics/pyfixest/issues/1148) | Poisson via GLM: deprecate `Fepois`, `pf.fepois()` calls `pf.feglm()` internally | Issue | `TBD` |
| — | Port HAC standard errors to Rust | `metrics` | `TBD` |
| — | Varying slopes syntax for demeaning | Idea | `TBD` |
| [#791](https://github.com/py-econometrics/pyfixest/issues/791) | IV with multiple endogenous variables | `metrics` | `TBD` |
| [PR #1099](https://github.com/py-econometrics/pyfixest/pull/1099) | Sensitivity analysis a la Cinelli & Hazlett (needs to be concluded) | PR | `TBD` |
| — | Anderson-Rubin confidence intervals for IV (robust to weak instruments) | `metrics` | `TBD` |
| — | Tutorial: shift-share IV with recentered instruments (Borusyak & Hull) | `metrics` | `TBD` |


---

## Setting Up the Dev Environment

PyFixest uses [pixi](https://pixi.sh) for environment management. To get started:

```bash
# 1. Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Clone and enter the repo
git clone https://github.com/py-econometrics/pyfixest.git
cd pyfixest

# 3. Install all dev dependencies
pixi install -e dev

# 4. Run Python with all dependencies available
pixi run -e dev python

# 5. Run tests
pixi run -e dev pytest tests/

# 6. Run linter + formatting checks (pre-commit hooks)
pixi run -e lint pre-commit
```

The `dev` environment includes pytest, rpy2 (for cross-validation against R's `fixest`), polars, duckdb, and other dev tools. Always use `pixi run -e dev` instead of bare `python` to ensure dependencies are available.

PyFixest also has a Rust extension (in `src/`) for performance-critical operations. Running `pixi install` handles the Rust build via maturin automatically.

For agents, see the concise command guide in [`AGENT_DEVELOPMENT.md`](AGENT_DEVELOPMENT.md) (env setup, tests, docs build, linting).

---

## Codebase Architecture

### Overview

```
pyfixest/
├── estimation/
│   ├── api/            # User-facing functional API: feols(), fepois(), feglm(), quantreg()
│   ├── FixestMulti_.py # Orchestrator: manages multi-model estimation
│   ├── models/         # Individual model classes: Feols, Fepois, Feiv, Felogit, ...
│   ├── formula/        # Formula parsing (Formula) and model matrix creation (ModelMatrix)
│   └── internals/      # Demeaning, solvers, vcov utilities, backend selection
├── core/               # Rust-backed core: demean(), crv1(), collinearity checks
├── did/                # Difference-in-differences estimators (DID2S, LP-DID, etc.)
├── report/             # etable(), coefplot(), summary visualization
└── utils/              # Data generating processes, helper functions
```

### Request Flow

**1. Functional API** (`estimation/api/feols.py`, `fepois.py`, etc.)

Users call `pf.feols("Y ~ X1 + X2 | f1 + f2", data)`. These functions:
- Validate inputs
- Create a `FixestMulti` instance
- Call `_prepare_estimation()` and `_estimate_all_models()`
- Return a single `Feols` object (single model) or `FixestMulti` (multiple models)

**2. FixestMulti** (`estimation/FixestMulti_.py`)

The orchestrator class. It:
- Parses the formula via `Formula.parse_to_dict()` to expand multi-estimation syntax (`sw()`, `csw()`, multiple depvars)
- Loops over all formula/split combinations in `_estimate_all_models()`
- For each model: instantiates the right model class (via a `model_map` dict), calls `prepare_model_matrix()` → `get_fit()` → `vcov()` → `get_inference()`
- Maintains a `lookup_demeaned_data` cache dict shared across models with the same fixed effects

**3. Model Classes** (`estimation/models/`)

- `Feols` — OLS (the base class, all others inherit or follow its pattern)
- `Fepois` — Poisson regression (IRLS/IWLS)
- `Feiv` — IV/2SLS
- `Felogit`, `Feprobit`, `Fegaussian` — GLM family
- `FeolsCompressed` — OLS on compressed/sufficient-statistics data
- `Quantreg` — Quantile regression

Each model class implements `prepare_model_matrix()`, `get_fit()`, and model-specific `vcov()` methods. `Feols` also provides shared post-estimation methods: `predict()`, `coef()`, `se()`, `confint()`, `wildboottest()`, `ritest()`, `ccv()`, `decompose()`, `fixef()`.

**4. Formula Parsing** (`estimation/formula/`)

- `Formula` (in `parse.py`) — A dataclass that parses fixest-style formula strings (`"Y1 + Y2 ~ X1 | sw(f1, f2) | X2 ~ Z1"`) into structured parts: dependent vars, exogenous vars, fixed effects, endogenous/instrument pairs. Handles `sw()`, `csw()`, `sw0()`, `csw0()`, `i()` interactions, and `^` interacted FEs. `parse_to_dict()` expands multi-estimation syntax into a dict of `{fixef_key: [Formula, ...]}`.
- `ModelMatrix` (in `model_matrix.py`) — Takes a `Formula` + data, uses `formulaic` to build numpy design matrices (Y, X, FE, instruments, weights). Tracks NA indices for cache alignment.

**5. Demeaning & Caching** (`estimation/internals/demean_.py`, `core/demean.py`)

The demeaning pipeline:
- `Feols.demean()` calls `demean_model()` (in `internals/demean_.py`)
- `demean_model()` checks a `lookup_demeaned_data` dict (keyed by `na_index` as frozenset) to see if variables were already demeaned for the same FE/missing-value pattern
- Cache hit: only demean the new variables not yet in the cache
- Cache miss: demean everything, store results
- The actual demeaning dispatches to one of several backends:
  - **Rust** (`core/demean.py` → compiled `.so`): alternating projections, called via PyO3
  - **Numba** (default): JIT-compiled alternating projections
  - **JAX**: GPU-capable alternating projections
  - **CuPy/SciPy**: sparse FWL-based approach for GPU or CPU

The Rust extension (`src/demean.rs`) also provides `crv1` (cluster-robust variance) and collinearity detection routines.

---

## Tests

Tests live in `tests/` and are run via `pixi run -e dev pytest tests/`.

### Cross-validation against R's `fixest` (`test_vs_fixest.py`)

The most important test file. It uses `rpy2` to call R's `fixest` package and compares coefficients, standard errors, t-stats, p-values, confidence intervals, R-squared, residuals, and predictions between PyFixest and R. Tests are heavily parametrized across:
- Formulas (OLS, Poisson, IV, GLM)
- Inference types (iid, hetero, CRV1)
- Weights (weighted / unweighted)
- Demeaner backends (numba, rust, cupy, scipy)
- Fixed effect types (float, int, categorical)
- With/without NAs, singletons, SSC adjustments

There's also `test_hac_vs_fixest.py` for HAC (Newey-West, Driscoll-Kraay) standard errors against R.

These tests require R + `fixest` installed (handled by the `dev` pixi environment). They are marked with `@pytest.mark.against_r_core`.

### Demeaner tests (`test_demean.py`)

Tests the demeaning internals directly:
- Correctness of `demean()` across backends
- `demean_model()` with/without fixed effects and weights
- **Caching**: verifies that `lookup_demeaned_data` is populated on first call and reused on subsequent calls with overlapping variables
- Convergence failure when `maxiter` is too low
- Integration test: `feols()` with custom `fixef_maxiter`

### Other notable test files

| File | What it tests |
|------|--------------|
| `test_formula_parse.py` | Formula parsing and multi-estimation expansion |
| `test_formulas.py` | Formula syntax (sw, csw, i(), interactions) |
| `test_collinearity.py` | Multicollinearity detection and variable dropping |
| `test_crv1.py` | Cluster-robust variance estimation (Rust impl) |
| `test_detect_singletons.py` | Singleton fixed effect detection/removal |
| `test_predict_resid_fixef.py` | predict(), resid(), fixef() methods |
| `test_did.py` | Difference-in-differences estimators |
| `test_wildboottest.py` | Wild (cluster) bootstrap inference |
| `test_quantreg.py` | Quantile regression |
| `test_feols_compressed.py` | Compressed/sufficient-statistics OLS |

---
