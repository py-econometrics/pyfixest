# PyFixest Sprint in Heilbronn


We're organizing a PyFixest development sprint in partnership with the [appliedAI Institute](https://appliedai-institute.de/) at their Heilbronn office.

**Dates:** March 4th–6th 2026.

**Interested in joining?** Reach out to [Alex](mailto:alexander-fischer1801@t-online.de) with a brief note about your background and motivation. We have some funding available to support student participation.

### What we're working on

Our main goals for the sprint:

- **Rust backend:** Finalize the port from Numba to Rust and deprecate the Numba dependency, with continued optimization of our core demeaning algorithm. We still have to port logic for HAC standard errors and randomisation inference.
- **GPU acceleration:** Continue building out JAX, CuPy, and potentially PyTorch backends (for Mac users), potentially re-implementing the [LSMR algorithm](https://web.stanford.edu/group/SOL/software/lsmr/LSMR-SISC-2011.pdf) by hand & run experiments on pre-conditioning
- **Internal refactor:** Introduce a cleaner class hierarchy with a proper base estimation class. Currently, all estimation classes inherit from `Feols`.
- **NumPy-style API:** Rewrite estimation classes (Feols, Fepois, etc.) to follow sklearn-style conventions. Users should be able to fit a regression model by passing `data`, `X` or `y` to `Feols`. If data is passed, a `Feols.from_formula` method creates the design matrix. Core functional estimation APIs (`feols()`, `fepois`, etc) remain as they are.
- **Clean Handling of Regression Weights**: The logic for weighted least squares (WLS) is currently a bit hard to follow, as dependent variable and design matrix are pre-multiplied with `np.sqrt(weights)`. By reading the code, it is not always immediately clear if `self._X` is already weighted, or not yet. This of course can lead to confusion and bugs, which we should strive to avoid =)


We're also hoping to make progress on:

- **Standalone demeaning package:** Spin out the demeaning algorithms into a lightweight, cross-language package
- **Varying slopes support:** Add varying slopes to the demeaning algorithm and extend the formula API
- **Narwhals integration:** Better support for running analyses with either pandas or polars
- **maketables cleanup:** Refactor the codebase and open PRs to third-party packages (doubleML, CausalPy, econML, etc.)
- **Documentation:** Add how-to guides and add a *statistical documentation* in which we explan the math behind all estimators
- **Instrumental Variables**: We want to support IV models with more than one endogenous variable, and implement a range of diagnostics.
- **moderndid contributions:** Port our DiD estimators (Gardner, local projections, Sun & Abraham) to the [moderndid](https://github.com/jordandeklerk/moderndid) package
- **Tests**: The `pyfixest` test suite has grown quite a lot over time, and resembles more of
a djungle than an english garden. It would be great to clean things up little bit. In particular, it would be lovely to move away from `rpy2`/ use modern features of it.

### What would be helpful

If you're excited about econometrics tooling and want to contribute, we'd love to have you. You don't need to be an expert in Rust or GPU programming. If you've been looking for a way to get started with open source, we'd be happy to help you make your first contributions.
