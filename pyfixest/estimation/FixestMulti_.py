"""Collect and post-process models produced by multiple estimation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pandas as pd

from pyfixest.estimation.config import EstimationConfig
from pyfixest.estimation.models._result_accessor_mixin import (
    ReportAccessorMixin,
    TidyColumnAccessors,
)
from pyfixest.estimation.plan_ import ParsedFormula
from pyfixest.typing import (
    ModelResult,
    QuantregVcovType,
    RegressionVcovType,
    VcovKwargs,
)


class FixestMulti(ReportAccessorMixin, TidyColumnAccessors):
    """Container for models produced by one multiple-estimation call.

    `feols()`, `fepois()`, `feglm()`, and `quantreg()` return this class when a
    formula, quantile list, `split`, or `fsplit` expands into multiple models.
    Public APIs first create an `EstimationConfig`; `parse_formula()` expands the
    call, and `runner.run_estimation()` fits each planned model while sharing
    compatible demeaning caches. `FixestMulti` is only the result container: it
    does not parse formulas, demean data, or fit models itself.

    Use `to_list()` to obtain every model or `fetch_model()` to select one. The
    reporting methods `summary()`, `etable()`, `coefplot()`, and `iplot()` apply
    to the complete collection.
    """

    def __init__(
        self,
        *,
        config: EstimationConfig,
        parsed: ParsedFormula,
        data: pd.DataFrame,
        context: Mapping[str, Any],
    ) -> None:
        """Initialize the result container used by the estimation runner.

        Parameters
        ----------
        config : EstimationConfig
            Immutable record of every option the public API requested.
        parsed : ParsedFormula
            Result of `plan_.parse_formula(config)`.
        data : pandas.DataFrame
            The input data after narwhals→pandas conversion, optional copy,
            and index reset.
        context : Mapping[str, Any]
            Captured evaluation scope (from `capture_context`).
        """
        self._config = config
        self._parsed = parsed
        self._data = data
        self._context = context

        self.all_fitted_models: dict[str, ModelResult] = {}

    def _report_models(self) -> list[ModelResult]:
        """Return every fitted model for the shared reporting wrappers."""
        return list(self.all_fitted_models.values())

    @property
    def _is_iv(self) -> bool:
        """Whether the call expanded into an IV model."""
        return self._parsed.is_iv

    @property
    def _is_multiple_estimation(self) -> bool:
        """Whether the call expanded into more than one model."""
        return self._parsed.is_multiple_estimation

    @property
    def FixestFormulaDict(self):
        """Parsed formula dict keyed by fixed-effects spec."""
        return self._parsed.formula_dict

    def to_list(self) -> list[ModelResult]:
        """Return all fitted models in estimation order.

        Returns
        -------
        list[ModelResult]
            Fitted OLS, IV, GLM, Poisson, or quantile-regression models.
        """
        return list(self.all_fitted_models.values())

    def vcov(
        self,
        vcov: RegressionVcovType | QuantregVcovType | dict[str, str],
        vcov_kwargs: VcovKwargs | None = None,
    ) -> FixestMulti:
        """
        Update inference for every fitted model in place.

        Parameters
        ----------
        vcov : RegressionVcovType, QuantregVcovType, or dict[str, str]
            Covariance estimator accepted by every contained model. Regression
            models support iid, heteroskedastic, clustered, NW, and DK inference;
            quantile models support iid, nid, heteroskedastic, or one-way CRV1.
        vcov_kwargs : VcovKwargs, optional
            HAC arguments such as `lag`, `time_id`, and `panel_id`.

        Returns
        -------
        FixestMulti
            This container with updated inference.
        """
        for fxst in self.all_fitted_models.values():
            cast(Any, fxst).vcov(vcov=vcov, vcov_kwargs=vcov_kwargs)
        return self

    def tidy(self) -> pd.DataFrame:
        """
        Combine every model's coefficient table into one tidy DataFrame.

        Returns
        -------
        pandas.DataFrame
            Coefficient statistics indexed by formula and coefficient name.
        """
        res = []
        for x in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[x]
            df = fxst.tidy().reset_index()
            df["fml"] = fxst._fml
            res.append(df)

        res_df = pd.concat(res, axis=0)
        res_df.set_index(["fml", "Coefficient"], inplace=True)

        return res_df

    def confint(self) -> pd.DataFrame:
        """
        Obtain confidence intervals for the fitted models.

        Returns
        -------
        pandas.DataFrame
            Lower and upper confidence bounds indexed by formula and coefficient.
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
        """
        Run a wild cluster bootstrap for all regressions in the Fixest object.

        Parameters
        ----------
        reps : int
            The number of bootstrap iterations to run.
        param : Union[str, None], optional
            A string of length one, containing the test parameter of interest.
            Default is None.
        cluster : Union[str, None], optional
            The name of the cluster variable. Default is None. If None, uses
            the `self._clustervar` attribute as the cluster variable. If the
            `self._clustervar` attribute is None, a heteroskedasticity-robust
            wild bootstrap is run.
        weights_type : str, optional
            The type of bootstrap weights. Either 'rademacher', 'mammen', 'webb',
            or 'normal'. Default is 'rademacher'.
        impose_null : bool, optional
            Should the null hypothesis be imposed on the bootstrap dgp, or not?
            Default is True.
        bootstrap_type : str, optional
            A string of length one. Allows choosing the bootstrap type to be run.
            Either '11', '31', '13', or '33'. Default is '11'.
        seed : Union[str, None], optional
            Option to provide a random seed. Default is None.
        k_adj: bool, optional
            Whether to adjust the original coefficients with the bootstrap distribution.
            Default is True.
        G_adj : bool, optional
            Whether to adjust standard errors for clustering in the bootstrap.
            Default is True.

        Returns
        -------
        pandas.DataFrame
            A pd.DataFrame with bootstrapped t-statistic and p-value.
            The index indicates which model the estimated statistic derives from.
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

    def fetch_model(self, i: int | str, print_fml: bool | None = True) -> ModelResult:
        """
        Fetch one fitted model by its zero-based position.

        Parameters
        ----------
        i : int or str
            The index of the model to fetch.
        print_fml : bool, optional
            Whether to print the formula of the model. Default is True.

        Returns
        -------
        ModelResult
            The selected OLS, IV, GLM, Poisson, or quantile result.
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
