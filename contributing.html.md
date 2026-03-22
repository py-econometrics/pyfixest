## Overview

Thanks for showing interest in contributing to `pyfixest`! We appreciate all
contributions and constructive feedback, whether that be reporting bugs, requesting
new features, or suggesting improvements to documentation.

If you'd like to get involved, but are not yet sure how, please feel free to send us an [email](alexander-fischer1801@t-online.de). Some familiarity with
either Python or econometrics will help, but you really don't need to be a `numpy` core developer or have published in [Econometrica](https://onlinelibrary.wiley.com/journal/14680262) =) We'd be more than happy to invest time to help you get started!

::: {.callout-note}
## PyFixest Sprint in Heilbronn

We're hosting a [PyFixest Sprint](pyfixest-sprint.md) with [AppliedAI](https://www.appliedai.de/) in late February/early March 2026. If you're interested in contributing to PyFixest in person, [learn more and get in touch](pyfixest-sprint.md)!
:::

For a comprehensive overview of the codebase architecture and internals, check out the [DeepWiki](https://deepwiki.com/py-econometrics/pyfixest). While not perfect and correct in all regards, we think it is a pretty good starting point to learn about the codebase!

## Reporting bugs

We use [GitHub issues](https://github.com/py-econometrics/pyfixest/issues) to track bugs. You can report a bug by opening a new issue or contribute to an existing issue if
related to the bug you are reporting.

Before creating a bug report, please check that your bug has not already been reported, and that your bug exists on the latest version of pyfixest. If you find a closed issue that seems to report the same bug you're experiencing, open a new issue and include a link to the original issue in your issue description.

Please include as many details as possible in your bug report. The information helps the maintainers resolve the issue faster.

## Suggesting enhancements

We use [GitHub issues](https://github.com/py-econometrics/pyfixest/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement) to track bugs and suggested enhancements. You can suggest an enhancement by opening a new feature request. Before creating an enhancement suggestion, please check that a similar issue does not already exist.

Please describe the behavior you want and why, and provide examples of how pyfixest would be used if your feature were added.

## Contributing to the codebase

### Setting up your local environment

First, fork the pyfixest GitHub repository, then clone your fork:

```{.bash .code-copy}
git clone https://github.com/<username>/pyfixest.git
cd pyfixest
```

### Package Management via `pixi`

`PyFixest` uses [pixi](https://pixi.sh/latest/) for package management. All configuration lives in `pyproject.toml` under `[tool.pixi.*]` sections.

To install `pixi`, follow the [installation instructions](https://pixi.sh/latest/#installation). Environments are installed automatically when you run a task (e.g. `pixi run test-py`). You do not need a global Rust installation -- Rust is installed via conda.

To see all available tasks and their descriptions, run:

```{.bash .code-copy}
pixi task list
```

### Running tests

```{.bash .code-copy}
# quick test suite (no R, no extended, no plots, no HAC)
pixi run test-py

# extended test suite
pixi run test-py-extended
```

To run tests that compare against R's `fixest`:

```{.bash .code-copy}
# tests against R's fixest (core R packages from conda-forge)
pixi run test-r-core

# tests against R's fixest (extra R packages from CRAN)
pixi run test-r-extended

# run test_vs_fixest.py specifically
pixi run test-r-fixest

# HAC standard error tests (runs with pinned thread counts)
pixi run test-r-hac

# run everything (all markers, all tests)
pixi run test-all
```

Pixi will prompt you to pick a Python version for the R environment — any option works.

### Linting

We use [prek](https://prek.j178.dev/) (a fast, Rust-based re-implementation of
pre-commit) to run linting and formatting hooks.

**Install prek globally** (recommended for contributors):

```{.bash .code-copy}
pixi global install prek
```

**Set up git hooks** (run once after cloning):

```{.bash .code-copy}
prek install
```

This installs prek as a git hook so checks run automatically on each commit.

**Run all hooks manually:**

```{.bash .code-copy}
pixi run lint
```

### Building the documentation

```{.bash .code-copy}
pixi run docs-build
pixi run docs-render
pixi run docs-preview
```

## Developer guide

This section explains the packaging setup, pixi environments, and automation
so you understand how the pieces fit together.

### Rust extension and automatic rebuilds

pyfixest includes a Rust extension (`pyfixest.core._core_impl`) built via
[maturin](https://www.maturin.rs/). The build system is declared in
`pyproject.toml` under `[build-system]`.

**Initial build:** The editable install (`pyfixest = { path = ".", editable = true }`)
triggers maturin automatically. No manual build step is needed.

**Rebuilds after `.rs` changes:** We use
[maturin-import-hook](https://github.com/PyO3/maturin-import-hook) to
automatically rebuild the Rust extension whenever you `import pyfixest` and
the `.rs` source files have changed. The hook is installed via the `setup`
task, which runs automatically for pixi tasks that import `pyfixest`,
including tests and docs builds. You never need to manually rebuild.

### Task dependencies and caching

Several tasks use pixi's `depends-on`, `inputs`, and `outputs` to avoid
redundant work:

- **`_setup`** (maturin import hook): installs the hook into the active
  environment before tasks that import `pyfixest`.
- **`_update-test-data`**: runs `Rscript tests/r_test_comparisons.R` to
  generate `tests/data/ritest_results.csv`. Cached via `inputs`/`outputs` --
  only re-runs if the R script changes or the CSV is missing.

Note: `test-r-extended` requires extra R packages not available on conda-forge.
Install them manually before running: `Rscript r_test_requirements.R`.

These tasks are prefixed with `_` so they are hidden from `pixi task list`.

You can skip dependency tasks with `--skip-deps` if needed:

```{.bash .code-copy}
pixi run --skip-deps test-r-core
```
