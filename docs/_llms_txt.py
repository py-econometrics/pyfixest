"""Rewrite the site's llms.txt as a curated index.

Quarto's ``website: llms-txt: true`` writes a Markdown copy of every page
(``*.llms.md``) and an ``llms.txt`` that lists all of them under a single
heading, in render order, with no descriptions and with file names as titles
for pages that have no ``title`` field. Quarto offers no way to describe, group
or omit entries, so this post-render script (``project: post-render`` in
``_quarto.yml``) keeps Quarto's header (site title and description) and
replaces the page list with the sections below.

Guides come first, then the user-facing API reference. Internals (fitted-state
data classes, formula parsing, small-sample utilities) are not listed one by
one; the reference index links to them. The script runs on full renders only;
on a partial render Quarto keeps the existing llms.txt.

When you add a page to the site, add it to a section below or to ``UNLISTED``.
A listed page that was not rendered fails the build; a rendered page outside
``reference/`` that is in neither list prints a warning.
"""

import os
import sys
from pathlib import Path

# (page path without extension, title, description)
SECTIONS = {
    "Start here": [
        (
            "pyfixest",
            "PyFixest overview",
            "Features, installation (CPU and GPU), a quickstart, benchmarks and citation.",
        ),
        (
            "getting-started",
            "A first regression with PyFixest",
            "Install PyFixest, fit a first fixed-effects regression with clustered errors and compare models.",
        ),
        (
            "skills",
            "Agent skill for PyFixest",
            "A ready-to-use skill file for coding agents that write PyFixest code.",
        ),
    ],
    "Tutorials": [
        (
            "tutorials/ols-fixed-effects",
            "OLS with fixed effects",
            "What fixed effects absorb, with twin-study and worker-firm (AKM) applications.",
        ),
        (
            "tutorials/formula-syntax",
            "Formula syntax",
            "Fixed effects, interactions, `i()`, IV and multiple-estimation operators in model formulas.",
        ),
        (
            "tutorials/instrumental-variables",
            "Instrumental variables",
            "2SLS with fixed effects, first-stage results and the effective F-statistic.",
        ),
        (
            "tutorials/regression-tables",
            "Regression tables",
            "`etable()` tables: labels, significance stars, kept and dropped variables, and output formats.",
        ),
        (
            "tutorials/difference-in-differences",
            "Difference-in-differences",
            "TWFE, event studies, DID2S and local projections under staggered adoption.",
        ),
        (
            "tutorials/quantile-regression",
            "Quantile regression",
            "`quantreg()`: solvers, the quantile process, inference and performance.",
        ),
    ],
    "How-to guides": [
        (
            "how-to/demeaner-backends",
            "Choose a demeaner backend",
            "Pick and configure the fixed-effects demeaning algorithm (MAP or LSMR, CPU or GPU).",
        ),
        (
            "how-to/marginaleffects",
            "Marginal effects and hypothesis tests",
            "Average marginal effects and linear and non-linear tests with the `marginaleffects` package.",
        ),
        (
            "how-to/anytime-valid-inference",
            "Anytime-valid inference",
            "SAVI p-values, e-values and confidence sequences for experiments that are monitored and stopped early.",
        ),
        (
            "how-to/regression_decomposition",
            "Regression decomposition",
            "Gelbach decomposition: how much each covariate explains of a change in a coefficient.",
        ),
        (
            "how-to/stata-2-pyfixest",
            "Translate Stata to PyFixest",
            "`reg` and `reghdfe` commands with robust and clustered errors, and their PyFixest equivalents.",
        ),
        (
            "how-to/panel_variance_reduction",
            "Variance reduction in A/B tests",
            "CUPED, regression adjustment and panel fixed-effects estimators to gain power.",
        ),
        (
            "how-to/migrate-fit-attributes",
            "Migrate fitted-model attributes from 0.60",
            "Where 0.60 private attributes live on fitted models in 0.70 (`fit.model`, `fit.options`, ...).",
        ),
    ],
    "Explanation": [
        (
            "explanation/difficult-fixed-effects",
            "When are fixed effects problems difficult?",
            "Why the structure of the fixed effects drives runtime, and how the demeaning algorithms compare.",
        ),
        (
            "explanation/ssc",
            "Small sample corrections",
            "The `ssc()` options and how they enter standard errors and degrees of freedom.",
        ),
        (
            "explanation/compare-fixest-pyfixest",
            "Does PyFixest match fixest?",
            "Intentional formula differences from R's fixest, and side-by-side OLS and Poisson results.",
        ),
    ],
    "API reference: estimation": [
        (
            "reference/index",
            "Function reference index",
            "Every documented function and class with a one-line summary, including internals.",
        ),
        (
            "reference/estimation.api.feols.feols",
            "pf.feols",
            "OLS and IV regression with high-dimensional fixed effects; all `vcov`, weight and solver options.",
        ),
        (
            "reference/estimation.api.fepois.fepois",
            "pf.fepois",
            "Poisson regression with fixed effects (PPML).",
        ),
        (
            "reference/estimation.api.feglm.feglm",
            "pf.feglm",
            "GLMs with fixed effects: logit, probit, Gaussian and Poisson families.",
        ),
        (
            "reference/estimation.api.quantreg.quantreg",
            "pf.quantreg",
            "Linear quantile regression for one or several quantiles.",
        ),
        (
            "reference/did.estimation.event_study",
            "pf.event_study",
            "Event-study and difference-in-differences estimation (TWFE, DID2S, saturated).",
        ),
        (
            "reference/did.estimation.did2s",
            "pf.did2s",
            "Gardner's two-stage difference-in-differences estimator.",
        ),
        (
            "reference/did.estimation.lpdid",
            "pf.lpdid",
            "Local-projections difference-in-differences estimator.",
        ),
        (
            "reference/did.saturated_twfe.SaturatedEventStudy",
            "SaturatedEventStudy",
            "Saturated event study with cohort-specific effect curves.",
        ),
        (
            "reference/did.visualize.panelview",
            "pf.panelview",
            "Plot treatment status, or the outcome, over time for each unit of a panel.",
        ),
    ],
    "API reference: fitted models": [
        (
            "reference/estimation.models.feols_.Feols",
            "Feols",
            "The fitted OLS model returned by `feols()`; every attribute and method of a fit.",
        ),
        (
            "reference/estimation.models.feiv_.Feiv",
            "Feiv",
            "The fitted IV (2SLS) model, with first-stage results and diagnostics.",
        ),
        (
            "reference/estimation.models.fepois_.Fepois",
            "Fepois",
            "The fitted Poisson model returned by `fepois()`.",
        ),
        (
            "reference/estimation.models.feglm_.Feglm",
            "Feglm",
            "The fitted GLM returned by `feglm()`.",
        ),
        (
            "reference/estimation.quantreg.quantreg_.Quantreg",
            "Quantreg",
            "The fitted quantile regression returned by `quantreg()`.",
        ),
        (
            "reference/estimation.FixestMulti_.FixestMulti",
            "FixestMulti",
            "Container for multiple estimations from one call; `to_list()`, `fetch_model()`, joint `tidy()`.",
        ),
    ],
    "API reference: post-estimation methods": [
        (
            "reference/estimation.models.feols_.Feols.tidy",
            "fit.tidy()",
            "Coefficient table as a DataFrame: estimates, standard errors, t-statistics, p-values and intervals.",
        ),
        (
            "reference/estimation.models.feols_.Feols.coef",
            "fit.coef()",
            "Estimated coefficients.",
        ),
        (
            "reference/estimation.models.feols_.Feols.se",
            "fit.se()",
            "Standard errors.",
        ),
        (
            "reference/estimation.models.feols_.Feols.tstat",
            "fit.tstat()",
            "t-statistics.",
        ),
        (
            "reference/estimation.models.feols_.Feols.pvalue",
            "fit.pvalue()",
            "p-values.",
        ),
        (
            "reference/estimation.models.feols_.Feols.confint",
            "fit.confint()",
            "Confidence intervals, pointwise or simultaneous.",
        ),
        (
            "reference/estimation.models.feols_.Feols.resid",
            "fit.resid()",
            "Residuals.",
        ),
        (
            "reference/estimation.models.feols_.Feols.vcov",
            "fit.vcov()",
            "Recompute inference with another covariance estimator (iid, heteroskedastic, clustered, HAC).",
        ),
        (
            "reference/estimation.models.feols_.Feols.predict",
            "fit.predict()",
            "Predictions on the estimation sample or new data, optionally with standard errors.",
        ),
        (
            "reference/estimation.models.feols_.Feols.fixef",
            "fit.fixef()",
            "Recover the estimated fixed-effect coefficients.",
        ),
        (
            "reference/estimation.models.feols_.Feols.wald_test",
            "fit.wald_test()",
            "Wald test of linear restrictions on the coefficients.",
        ),
        (
            "reference/estimation.models.feols_.Feols.wildboottest",
            "fit.wildboottest()",
            "Wild (cluster) bootstrap test of a single coefficient.",
        ),
        (
            "reference/estimation.models.feols_.Feols.ritest",
            "fit.ritest()",
            "Randomization inference test of a single coefficient.",
        ),
        (
            "reference/estimation.models.feols_.Feols.ccv",
            "fit.ccv()",
            "Causal cluster variance estimator (Abadie et al., 2023).",
        ),
        (
            "reference/estimation.models.feols_.Feols.decompose",
            "fit.decompose()",
            "Gelbach decomposition of a coefficient across covariates.",
        ),
        (
            "reference/estimation.models.feols_.Feols.update",
            "fit.update()",
            "Update OLS coefficients with new observations without refitting.",
        ),
        (
            "reference/estimation.models.feols_.Feols.evalue",
            "fit.evalue()",
            "SAVI e-values for anytime-valid inference.",
        ),
        (
            "reference/estimation.models.feols_.Feols.pvalue_savi",
            "fit.pvalue_savi()",
            "SAVI sequential p-values for anytime-valid inference.",
        ),
        (
            "reference/estimation.post_estimation.multcomp.bonferroni",
            "pf.bonferroni",
            "Bonferroni-adjusted p-values across models.",
        ),
        (
            "reference/estimation.post_estimation.multcomp.rwolf",
            "pf.rwolf",
            "Romano-Wolf adjusted p-values across models (bootstrap).",
        ),
        (
            "reference/estimation.post_estimation.multcomp.wyoung",
            "pf.wyoung",
            "Westfall-Young adjusted p-values across models (bootstrap).",
        ),
    ],
    "API reference: tables and plots": [
        (
            "reference/report.etable",
            "pf.etable",
            "Regression table for one or more models as Great Tables, DataFrame, Markdown, LaTeX, Typst or Word.",
        ),
        (
            "reference/report.summary",
            "pf.summary",
            "Print a text summary of one or more fitted models.",
        ),
        (
            "reference/report.coefplot",
            "pf.coefplot",
            "Coefficient plot with confidence intervals for one or more models.",
        ),
        (
            "reference/report.iplot",
            "pf.iplot",
            "Plot coefficients of `i()` interactions, such as event-study estimates.",
        ),
        (
            "reference/report.qplot",
            "pf.qplot",
            "Plot coefficients across quantiles from `quantreg()`.",
        ),
    ],
    "API reference: example data": [
        (
            "reference/utils.utils.get_data",
            "pf.get_data",
            "Random example data with covariates, fixed effects `f1` to `f3` and an instrument, used throughout the docs.",
        ),
        (
            "reference/utils.dgps.get_twin_data",
            "pf.get_twin_data",
            "Twin-study data on the returns to education.",
        ),
        (
            "reference/utils.dgps.get_worker_panel",
            "pf.get_worker_panel",
            "Worker-firm panel with two-way fixed effects.",
        ),
        (
            "reference/utils.dgps.get_ivf_data",
            "pf.get_ivf_data",
            "IV data for the motherhood penalty with an IVF instrument.",
        ),
        (
            "reference/utils.dgps.get_bartik_data",
            "pf.get_bartik_data",
            "Shift-share (Bartik) IV data on immigration and wages.",
        ),
        (
            "reference/utils.dgps.get_encouragement_data",
            "pf.get_encouragement_data",
            "IV data from an A/B encouragement design.",
        ),
        (
            "reference/utils.dgps.get_motherhood_event_study_data",
            "pf.get_motherhood_event_study_data",
            "Fertility-timing panel for event studies of the motherhood penalty.",
        ),
    ],
    "Optional": [
        (
            "textbook-replications/brave_true",
            "Causal Inference for the Brave and True",
            "Panel-data and difference-in-differences chapters replicated with PyFixest.",
        ),
        (
            "textbook-replications/mixtape",
            "Causal Inference: The Mixtape",
            "Panel-data and difference-in-differences chapters replicated with PyFixest.",
        ),
        (
            "textbook-replications/replicating-the-effect",
            "The Effect",
            "Regression, fixed-effects and difference-in-differences chapters replicated with PyFixest.",
        ),
        (
            "textbook-replications/resources",
            "Learning econometrics with PyFixest",
            "Textbooks, courses and notebooks that teach with PyFixest.",
        ),
        (
            "explanation/blog-posts",
            "Case studies and blog posts",
            "Uses of PyFixest in industry and research, written up elsewhere.",
        ),
        (
            "changelog",
            "Changelog",
            "Release notes, including deprecations and breaking changes.",
        ),
        (
            "contributing",
            "Contributing",
            "Development setup, tests and how to propose changes.",
        ),
        (
            "acknowledgements",
            "Acknowledgements",
            "Software and papers PyFixest builds on.",
        ),
    ],
}

# Rendered pages deliberately left out: the redirect at the site root and
# section landing pages that only link to pages listed above.
UNLISTED = {
    "index",
    "tutorials/index",
    "how-to/index",
    "textbook-replications/index",
}


def main() -> None:
    """Replace the page list of the llms.txt Quarto wrote."""
    if not os.environ.get("QUARTO_PROJECT_RENDER_ALL"):
        return
    output_dir = Path(os.environ["QUARTO_PROJECT_OUTPUT_DIR"])
    llms_txt = output_dir / "llms.txt"

    generated = llms_txt.read_text(encoding="utf-8")
    header, found, _ = generated.partition("## Pages\n")
    if not found:
        sys.exit(f"{llms_txt}: no '## Pages' heading; did Quarto's format change?")

    listed = {path for pages in SECTIONS.values() for path, _, _ in pages}
    missing = sorted(
        path for path in listed if not (output_dir / f"{path}.llms.md").exists()
    )
    if missing:
        sys.exit(
            "docs/_llms_txt.py lists pages that were not rendered: "
            + ", ".join(missing)
        )

    rendered = {
        p.relative_to(output_dir).as_posix().removesuffix(".llms.md")
        for p in output_dir.rglob("*.llms.md")
    }
    unlisted = sorted(
        path
        for path in rendered - listed - UNLISTED
        if not path.startswith("reference/")
    )
    if unlisted:
        print(
            "Warning: pages missing from llms.txt; add them to docs/_llms_txt.py: "
            + ", ".join(unlisted),
            file=sys.stderr,
        )

    lines = [header.rstrip("\n"), ""]
    for heading, pages in SECTIONS.items():
        lines += [f"## {heading}", ""]
        lines += [
            f"- [{title}]({path}.llms.md): {description}"
            for path, title, description in pages
        ]
        lines.append("")
    llms_txt.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
