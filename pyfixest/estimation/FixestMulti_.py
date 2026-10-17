from __future__ import annotations

import warnings
from collections.abc import Mapping

import pandas as pd

from pyfixest.estimation.formula.parse import Formula as FixestFormula
from pyfixest.estimation.models._result_accessor_mixin import TidyColumnAccessors
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois


class FixestMulti(TidyColumnAccessors):
    """Fitted-model container returned by a multiple-estimation call.

    Access as ``fits = pf.feols(...)`` or ``fits = pf.quantreg(...)`` when
    the call requests several fitted models. Use the estimation functions
    rather than constructing this container directly.

    Returned instead of a single model when the formula uses multiple estimation
    syntax: several dependent variables, the stepwise operators (`sw`, `sw0`,
    `csw`, `csw0`, `mvsw`), stepwise fixed effects, or a `split` / `fsplit`
    sample split. Reporting methods such as `etable()`, `summary()` and
    `coefplot()` apply to all fitted models at once. `fetch_model()` returns an
    individual model. `quantreg()` with a list of quantiles also returns a
    `FixestMulti`, with one model per quantile.

    Attributes
    ----------
    all_fitted_models : dict[str, Feols]
        The fitted models, keyed by their formula, with the sample or the
        quantile appended for split fits and quantile processes. Values are
        `Feols` objects or one of its subclasses (`Feiv`, `Feglm`, `Fepois`,
        `Quantreg`).

    Examples
    --------
    ```{python}
    import pyfixest as pf

    data = pf.get_data()
    fits = pf.feols("Y ~ X1 | csw0(f1, f2)", data)

    pf.etable(fits)
    ```

    Retrieve one of the fitted models to work with it on its own:

    ```{python}
    fit = fits.fetch_model(0, print_fml=False)
    fit.tidy()
    ```
    """

    def __init__(
        self, *, formula_dict: Mapping[str | None, list[FixestFormula]]
    ) -> None:
        """Create an empty container; the estimation runner adds the fitted models.

        Parameters
        ----------
        formula_dict : Mapping[str | None, list[FixestFormula]]
            The parsed formulas keyed by fixed-effects spec, kept only for the
            deprecated `FixestFormulaDict` attribute.
        """
        self._formula_dict = formula_dict

        self.all_fitted_models: dict[str, Feols | Fepois | Feiv] = {}

    @property
    def FixestFormulaDict(self) -> Mapping[str | None, list[FixestFormula]]:
        """Return the deprecated mapping of parsed model formulas.

        Use ``fit.model.parsed_formula`` on the models from ``fits.to_list()``
        instead. Reading this property emits ``FutureWarning``.

        Returns
        -------
        Mapping[str | None, list[Formula]]
            Parsed formulas keyed by fixed-effect specification.

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        warnings.warn(
            "`FixestFormulaDict` is deprecated and will be removed in a future "
            "release. Use `fit.model.parsed_formula` on the models returned by "
            "`to_list()` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self._formula_dict

    def summary(self, **kwargs):
        """Print a model summary.

        Parameters
        ----------
        **kwargs
            Arguments forwarded to ``pf.summary()`` with this result as ``models``.

        Returns
        -------
        None
            Prints the summary to standard output.

        See Also
        --------
        [summary()](/reference/report.summary.qmd),
        [fitted-result overview](/reference-guide/fitted-results.qmd).
        """
        from pyfixest.report import summary

        return summary(models=self, **kwargs)

    def etable(self, **kwargs):
        """Create a regression table.

        Parameters
        ----------
        **kwargs
            Arguments forwarded to ``pf.etable()`` with this result as ``models``.

        Returns
        -------
        pandas.DataFrame, GT, Document, str, or None
            Output selected by ``type``: table, formatted `DataFrame`, document,
            or text. Markdown output is printed and returns `None`. See the
            canonical ``pf.etable()`` page for formats and forwarded options.

        See Also
        --------
        [etable()](/reference/report.etable.qmd),
        [fitted-result overview](/reference-guide/fitted-results.qmd).
        """
        from pyfixest.report import etable

        return etable(models=self, **kwargs)

    def coefplot(self, **kwargs):
        """Plot model coefficients.

        Parameters
        ----------
        **kwargs
            Arguments forwarded to ``pf.coefplot()`` with this result as ``models``.

        Returns
        -------
        matplotlib.figure.Figure or lets_plot.plot.core.PlotSpec
            Plot object from the selected backend. See the canonical plotting
            page for arguments and save examples.

        See Also
        --------
        [coefplot()](/reference/report.coefplot.qmd),
        [fitted-result overview](/reference-guide/fitted-results.qmd).
        """
        from pyfixest.report import coefplot

        return coefplot(models=self, **kwargs)

    def iplot(self, **kwargs):
        """Plot coefficients generated by `i()` terms.

        Parameters
        ----------
        **kwargs
            Arguments forwarded to ``pf.iplot()`` with this result as ``models``.

        Returns
        -------
        matplotlib.figure.Figure or lets_plot.plot.core.PlotSpec
            Plot object from the selected backend. See the canonical plotting
            page for arguments and save examples.

        See Also
        --------
        [iplot()](/reference/report.iplot.qmd),
        [fitted-result overview](/reference-guide/fitted-results.qmd).
        """
        from pyfixest.report import iplot

        return iplot(models=self, **kwargs)

    def to_list(self) -> list[Feols | Fepois | Feiv]:
        """Return a list of all fitted models.

        Returns
        -------
        list
            A list in the insertion order of ``all_fitted_models``: `Feols` objects or one of its
            subclasses (`Feiv`, `Feglm`, `Fepois`, `Quantreg`).

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        return list(self.all_fitted_models.values())

    def vcov(
        self,
        vcov: str | dict[str, str],
        vcov_kwargs: dict[str, str | int] | None = None,
    ):
        """Update the inference of all fitted models.

        Calls `vcov()` on every model in the container, replacing its
        variance-covariance matrix. See
        [Feols.vcov()](/reference/estimation.models.feols_.Feols.vcov.qmd) for
        the options and the per-model restrictions.

        Parameters
        ----------
        vcov : str or dict[str, str]
            A string or dictionary specifying the type of variance-covariance
            matrix to use for inference. A string can be one of `iid`,
            `hetero`, `HC1`, `HC2`, `HC3`, `NW` or `DK` (`nid` for quantile
            regression). A dictionary has the format `{"CRV1": "clustervar"}`
            for CRV1 inference or `{"CRV3": "clustervar"}` for CRV3 inference.
        vcov_kwargs : dict[str, str | int], optional
            Additional arguments for the HAC estimators: "lag", "`time_id`" and
            "`panel_id`".

        Returns
        -------
        FixestMulti
            The same object, with updated inference.

        Notes
        -----
        Mutates every child model in order. If a later model rejects the
        requested estimator, earlier models may already have updated inference.
        All child models must satisfy the support and storage requirements of
        ``fit.vcov()``.

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd),
        [model storage](/reference-guide/model-storage.qmd).
        """
        for fxst in self.all_fitted_models.values():
            fxst.vcov(vcov=vcov, vcov_kwargs=vcov_kwargs)
        return self

    def tidy(self) -> pd.DataFrame:
        """Return the results of all fitted models as one tidy data frame.

        Returns
        -------
        pandas.DataFrame
            A data frame indexed by the levels `fml` (the model formula) and
            `Coefficient`, with the columns `Estimate`, `Std. Error`,
            `t value`, `Pr(>|t|)`, `2.5%` and `97.5%` (the bounds of the 95%
            confidence interval). Split fits add a `Sample` column.

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        res = []
        for x in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[x]
            df = fxst.tidy().reset_index()
            df["fml"] = fxst.model.formula
            res.append(df)

        res_df = pd.concat(res, axis=0)
        res_df.set_index(["fml", "Coefficient"], inplace=True)

        return res_df

    def confint(self) -> pd.DataFrame:
        """Obtain 95% confidence intervals for the fitted models.

        The level is fixed at 95%. For other levels, call `confint(alpha=...)`
        on the individual models from `to_list()`.

        Returns
        -------
        pandas.DataFrame
            A data frame with the columns `2.5%` and `97.5%`, indexed by the
            levels `fml` and `Coefficient`.

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        return self.tidy()[["2.5%", "97.5%"]]

    def wildboottest(
        self,
        reps: int,
        cluster: str | None = None,
        param: str | None = None,
        weights_type: str = "rademacher",
        impose_null: bool = True,
        bootstrap_type: str = "11",
        seed: int | None = None,
        k_adj: bool = True,
        G_adj: bool = True,
    ) -> pd.DataFrame:
        """Run a wild (cluster) bootstrap test for all fitted models.

        See [Feols.wildboottest()](/reference/estimation.models.feols_.Feols.qmd)
        for the options and the supported models.

        Parameters
        ----------
        reps : int
            The number of bootstrap iterations to run.
        cluster : str, optional
            The name of the cluster variable. Default is `None`. If `None`, uses
            each model's cluster variable. If a model is not clustered, a
            heteroskedasticity-robust wild bootstrap is run.
        param : str
            The name of the coefficient to test. Required in practice: with
            the default `None`, the test is not meaningful.
        weights_type : str, optional
            The type of bootstrap weights: `rademacher`, `mammen`, `webb` or
            'norm'. The heteroskedastic bootstrap supports only `rademacher`
            and 'norm'. Default is `rademacher`.
        impose_null : bool, optional
            Should the null hypothesis be imposed on the bootstrap dgp, or not?
            Default is `True`.
        bootstrap_type : str, optional
            The bootstrap type: `'11'` or `'31'`; the heteroskedastic bootstrap
            also accepts '21'. Default is `'11'`.
        seed : int, optional
            Option to provide a random seed. Default is `None`.
        k_adj : bool, optional
            Whether to apply the small sample adjustment for the number of
            observations and covariates. Only used by the cluster bootstrap.
            Default is `True`.
        G_adj : bool, optional
            Whether to apply the small sample adjustment for the number of
            clusters. Only used by the cluster bootstrap. Default is `True`.

        Returns
        -------
        pandas.DataFrame
            A data frame with the bootstrapped t-statistic and p-value per
            model, indexed by the model formula.

        Notes
        -----
        This method currently raises a `ValueError` when it assembles the
        results. Until that is fixed, call `wildboottest()` on the individual
        models from `to_list()`.

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        res_df = pd.DataFrame()
        for x in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[x]

            boot_res = fxst.wildboottest(
                reps,
                cluster,
                param,
                weights_type,
                impose_null,
                bootstrap_type,
                seed,
                k_adj,
                G_adj,
            )

            pvalue = boot_res["Pr(>|t|)"]
            tstat = boot_res["t value"]

            result_dict = {
                "fml": x,
                "param": param,
                "t value": tstat,
                "Pr(>|t|)": pvalue,
            }

            res_df = pd.concat([res_df, pd.DataFrame(result_dict)], axis=1)

        res_df = res_df.T.set_index("fml")

        return res_df

    def fetch_model(
        self, i: int | str, print_fml: bool | None = True
    ) -> Feols | Fepois:
        """Return one fitted model by its position.

        Parameters
        ----------
        i : int or str
            The position of the model in `all_fitted_models`, starting at 0.
            A string must hold an integer, e.g. `"0"`.
        print_fml : bool, optional
            Whether to print the formula of the model (`Model: ...`). Default
            is `True`.

        Returns
        -------
        Feols
            The fitted model: a `Feols` object or one of its subclasses.

        See Also
        --------
        [fitted-result overview](/reference-guide/fitted-results.qmd),
        [model support](/reference-guide/model-support.qmd).
        """
        if isinstance(i, str):
            i = int(i)

        keys = list(self.all_fitted_models.keys())
        if i >= len(keys):
            raise IndexError(f"Index {i} is larger than the number of fitted models.")
        key = keys[i]
        if print_fml:
            print("Model: ", key)
        model = self.all_fitted_models[key]
        return model
