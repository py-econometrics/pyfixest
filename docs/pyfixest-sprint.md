# PyFixest Sprint in Heilbronn


We're organizing a PyFixest development sprint in partnership with the [appliedAI Institute](https://appliedai-institute.de/) at their Heilbronn office. This is a chance to help shape the future of econometrics software in Python, and to work alongside PyFixest's core development team and AppliedAI's engineers for a few focused days of coding.

**Dates:** March 4th–6th 2026.

**Interested in joining?** Reach out to [Alex](mailto:alexander-fischer1801@t-online.de) with a brief note about your background and motivation. We have some funding available to support student participation.

### What we're working on

Our main goals for the sprint:

- **Rust backend:** Finalize the port from Numba to Rust and deprecate the Numba dependency, with continued optimization of our core demeaning algorithm
- **GPU acceleration:** Continue building out JAX, CuPy, and potentially PyTorch backends, potentially re-implementing the [LSMR algorithm](https://web.stanford.edu/group/SOL/software/lsmr/LSMR-SISC-2011.pdf) by hand
- **Internal refactor:** Introduce a cleaner class hierarchy with a proper base estimation class
- **NumPy-style API:** Rewrite estimation classes (Feols, Fepois, etc.) to follow sklearn-style conventions


We're also hoping to make progress on:

- **Standalone demeaning package:** Spin out the demeaning algorithms into a lightweight, cross-language package
- **Varying slopes support:** Add varying slopes to the demeaning algorithm and extend the formula API
- **Narwhals integration:** Better support for running analyses with either pandas or polars
- **maketables cleanup:** Refactor the codebase and open PRs to third-party packages (doubleML, CausalPy, econML, etc.)
- **moderndid contributions:** Port our DiD estimators (Gardner, local projections, Sun & Abraham) to the moderndid package
- **Documentation:** Add how-to guides and add a *statistical documentation* in which we explan the math behind all estimators

### What would be helpful

If you're excited about econometrics tooling and want to contribute, we'd love to have you. You don't need to be an expert in Rust or GPU programming—the AppliedAI team brings deep experience in both, so this is a great opportunity to learn. And if you've been looking for a way to get started with open source, we'd be happy to help you make your first contributions.
