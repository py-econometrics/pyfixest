from __future__ import annotations

import functools
from collections.abc import Mapping
from importlib import import_module
from typing import Any

import pandas as pd

from pyfixest.estimation.config import EstimationConfig
from pyfixest.estimation.models._result_accessor_mixin import TidyColumnAccessors
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.feols_ import (
    Feols,
    _check_vcov_input,
    _deparse_vcov_input,
)
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.plan_ import ParsedFormula


class FixestMulti(TidyColumnAccessors):
    """Results container holding every model fitted by one public-API call."""

    def __init__(
        self,
        *,
        config: EstimationConfig,
        parsed: ParsedFormula,
        data: pd.DataFrame,
        context: Mapping[str, Any],
    ) -> None:
        """.

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

        self.all_fitted_models: dict[str, Feols | Fepois | Feiv] = {}

        # set functions inherited from other modules
        _module = import_module("pyfixest.report")
        _tmp = _module.coefplot
        self.coefplot = functools.partial(_tmp, models=self.all_fitted_models.values())
        self.coefplot.__doc__ = _tmp.__doc__
        _tmp = _module.iplot
        self.iplot = functools.partial(_tmp, models=self.all_fitted_models.values())
        self.iplot.__doc__ = _tmp.__doc__
        _tmp = _module.summary
        self.summary = functools.partial(_tmp, models=self.all_fitted_models.values())
        self.summary.__doc__ = _tmp.__doc__
        _tmp = _module.etable
        self.etable = functools.partial(_tmp, models=self.all_fitted_models.values())
        self.etable.__doc__ = _tmp.__doc__

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

    def to_list(self) -> list[Feols | Fepois | Feiv]:
        """
        Return a list of all fitted models.

        Parameters
        ----------
            None

        Returns
        -------
            A list of all fitted models of types Feols or Fepois.
        """
        return list(self.all_fitted_models.values())

    def vcov(
        self,
        vcov: str | dict[str, str],
        vcov_kwargs: dict[str, str | int] | None = None,
    ):
        """
        Update regression inference "on the fly".

        By calling vcov() on a "Fixest" object, all inference procedures applied
        to the "Fixest" object are replaced with the variance-covariance matrix
        specified via the method.

        Parameters
        ----------
        vcov : Union[str, dict[str, str]])
            A string or dictionary specifying the type of variance-covariance
            matrix to use for inference.
            - If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3".
            - If a dictionary, it should have the format {"CRV1": "clustervar"}
            for CRV1 inference or {"CRV3": "clustervar"} for CRV3 inference.
        vcov_kwargs : Optional[dict[str, any]]
             Additional keyword arguments for the variance-covariance matrix.

        Returns
        -------
            An instance of the "Fixest" class with updated inference.
        """
        for model in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[model]
            _data = fxst._data

            _check_vcov_input(vcov=vcov, vcov_kwargs=vcov_kwargs, data=_data)
            (
                fxst._vcov_type,
                fxst._vcov_type_detail,
                _,
                _,
            ) = _deparse_vcov_input(vcov, False, False)

            fxst.vcov(vcov=vcov, vcov_kwargs=vcov_kwargs)
            fxst.get_inference()

        return self

    def tidy(self) -> pd.DataFrame:
        """
        Return the results of an estimation using `feols()` as a tidy Pandas DataFrame.

        Returns
        -------
        pandas.DataFrame or str
                A tidy DataFrame with the following columns:
                - fml: the formula used to generate the results
                - Coefficient: the names of the coefficients
                - Estimate: the estimated coefficients
                - Std. Error: the standard errors of the estimated coefficients
                - t value: the t-values of the estimated coefficients
                - Pr(>|t|): the p-values of the estimated coefficients
                - 2.5%: the lower bound of the 95% confidence interval
                - 97.5%: the upper bound of the 95% confidence interval
                If `type` is set to "markdown", the resulting DataFrame will be
                returned as a markdown-formatted string with three decimal places.
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
        pandas.Series
            A pd.Series with coefficient names and confidence intervals.
            The key indicates which models the estimated statistic derives from.
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
        B : int
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

    def fetch_model(
        self, i: int | str, print_fml: bool | None = True
    ) -> Feols | Fepois:
        """
        Fetch a model of class Feols from the Fixest class.

        Parameters
        ----------
        i : int or str
            The index of the model to fetch.
        print_fml : bool, optional
            Whether to print the formula of the model. Default is True.

        Returns
        -------
            A Feols object.
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
