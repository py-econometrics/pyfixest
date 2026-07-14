import functools
import warnings
from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyfixest.errors import EmptyVcovError

if TYPE_CHECKING:
    from pyfixest.estimation.internals.families import InferenceDist
from pyfixest.estimation.internals.literals import (
    InferenceType,
    _validate_literal_argument,
)
from pyfixest.utils.dev_utils import _select_coefnames_and_indices
from pyfixest.utils.utils import simultaneous_crit_val


class TidyColumnAccessors:
    """Mixin: derive `coef/se/tstat/pvalue` from `tidy()` data frame."""

    def tidy(self, *args, **kwargs) -> pd.DataFrame:
        """Tidy DataFrame of results. Implemented by the host class."""
        raise NotImplementedError

    def coef(self) -> pd.Series:
        """Estimated coefficients as a pandas Series."""
        return self.tidy()["Estimate"]

    def se(self) -> pd.Series:
        """Coefficient standard errors as a pandas Series."""
        return self.tidy()["Std. Error"]

    def tstat(self) -> pd.Series:
        """Coefficient t-statistics as a pandas Series."""
        return self.tidy()["t value"]

    def pvalue(self) -> pd.Series:
        """Coefficient p-values as a pandas Series."""
        return self.tidy()["Pr(>|t|)"]


class ResultAccessorMixin(TidyColumnAccessors):
    """Mixin providing result-accessor methods for fitted models."""

    # Type declarations for attributes provided by the host class (Feols).
    _vcov: np.ndarray
    _beta_hat: np.ndarray
    _se: np.ndarray
    _tstat: np.ndarray
    _pvalue: np.ndarray
    _conf_int: np.ndarray
    _u_hat: np.ndarray
    _weights: np.ndarray
    _Y: np.ndarray
    _Y_untransformed: pd.Series
    _coefnames: list[str]
    _method: str
    _drop_intercept: bool
    _has_fixef: bool
    _has_weights: bool
    _is_iv: bool
    _k_fe: pd.Series
    _N: int
    _k: int
    _df_t: int
    _inference_dist: "InferenceDist"
    _rmse: float
    _r2: float
    _adj_r2: float
    _r2_within: float
    _adj_r2_within: float
    _vcov_type: str

    def _bind_report_methods(self):
        """Bind summary, coefplot, iplot, and etable from pyfixest.report as instance methods."""
        _module = import_module("pyfixest.report")

        _tmp = _module.summary
        self.summary = functools.partial(_tmp, models=[self])
        self.summary.__doc__ = _tmp.__doc__

        _tmp = _module.coefplot
        self.coefplot = functools.partial(_tmp, models=[self])
        self.coefplot.__doc__ = _tmp.__doc__

        _tmp = _module.iplot
        self.iplot = functools.partial(_tmp, models=[self])
        self.iplot.__doc__ = _tmp.__doc__

        _tmp = _module.etable
        self.etable = functools.partial(_tmp, models=[self])
        self.etable.__doc__ = _tmp.__doc__

    def evalue(
        self,
        R: np.ndarray | None = None,
        q: float | np.ndarray | None = None,
        mixture_precision: float = 1.0,
    ) -> pd.Series | float:
        """Compute coefficient-wise or joint SAVI e-values.

        Parameters
        ----------
        R : np.ndarray, optional
            Restriction matrix. If omitted, returns one e-value per coefficient.
        q : float or np.ndarray, optional
            Value of the restriction under the null. Defaults to zero.
        mixture_precision : float, optional
            Positive mixture precision fixed before sequential monitoring.
            Defaults to 1. For coefficient-wise inference, use
            `pyfixest.estimation.post_estimation.savi.optimal_mixture_precision()`
            to minimize confidence-sequence width at a target sample size.

        Returns
        -------
        pd.Series or float
            Coefficient-wise e-values, or a scalar for a joint restriction.

        Notes
        -----
        SAVI currently supports unweighted, non-IV `feols` models without
        absorbed fixed effects. The covariance estimator must be iid or
        heteroskedasticity robust (`hetero`, `HC1`, `HC2`, or `HC3`).

        Examples
        --------
        ```{python}
        import numpy as np
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2", data=data, vcov="hetero")
        fit.evalue()

        R = np.array([[0.0, 1.0, -1.0]])
        fit.evalue(R=R)
        ```
        """
        from pyfixest.estimation.post_estimation.savi import _evalue

        return _evalue(
            model=self,
            R=R,
            q=q,
            mixture_precision=mixture_precision,
        )

    def sequential_pvalue(
        self,
        R: np.ndarray | None = None,
        q: float | np.ndarray | None = None,
        mixture_precision: float = 1.0,
    ) -> pd.Series | float:
        """Compute coefficient-wise or joint SAVI sequential p-values.

        The sequential-p-value analogue of `evalue`, returning
        `min(1, 1 / e_value)`. See `evalue` for the `R`, `q`, and
        `mixture_precision` arguments and the supported-model restrictions.

        Returns
        -------
        pd.Series or float
            Coefficient-wise sequential p-values, or a scalar for a joint
            restriction.

        Examples
        --------
        ```{python}
        import numpy as np
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2", data=data, vcov="HC1")
        fit.sequential_pvalue()
        fit.sequential_pvalue(R=np.array([[0.0, 1.0, -1.0]]))
        ```
        """
        from pyfixest.estimation.post_estimation.savi import _sequential_pvalue

        return _sequential_pvalue(
            model=self,
            R=R,
            q=q,
            mixture_precision=mixture_precision,
        )

    def get_inference(self, alpha: float = 0.05) -> None:
        """
        Compute standard errors, t-statistics, and p-values for the regression model.

        Parameters
        ----------
        alpha : float, optional
            The significance level for confidence intervals. Defaults to 0.05, which
            produces a 95% confidence interval.

        Returns
        -------
        None

        Details
        -------
        relevant fixest functions:
        - fixest_CI_factor: https://github.com/lrberge/fixest/blob/5523d48ef4a430fa2e82815ca589fc8a47168fe7/R/miscfuns.R#L5614
        -
        """
        if len(self._vcov) == 0:
            raise EmptyVcovError()

        self._se = np.sqrt(np.diagonal(self._vcov))
        self._tstat = self._beta_hat / self._se
        self._pvalue = self._inference_dist.pvalue(self._tstat, self._df_t)
        z = self._inference_dist.crit_val(alpha, self._df_t)

        z_se = z * self._se
        self._conf_int = np.array([self._beta_hat - z_se, self._beta_hat + z_se])

    def get_performance(self) -> None:
        """
        Get Goodness-of-Fit measures.

        Compute multiple additional measures commonly reported with linear
        regression output, including R-squared and adjusted R-squared. Note that
        variables with the suffix _within use demeaned dependent variables Y,
        while variables without do not or are invariant to demeaning.

        Returns
        -------
        None

        Creates the following instances:
        - r2 (float): R-squared of the regression model.
        - adj_r2 (float): Adjusted R-squared of the regression model.
        - r2_within (float): R-squared of the regression model, computed on
        demeaned dependent variable.
        - adj_r2_within (float): Adjusted R-squared of the regression model,
        computed on demeaned dependent variable.
        """
        Y_within = self._Y
        Y = self._Y_untransformed.to_numpy()

        has_intercept = not self._drop_intercept

        if self._has_fixef:
            k_fe = np.sum(self._k_fe - 1) + 1
            adj_factor = (self._N - has_intercept) / (self._N - self._k - k_fe)
            adj_factor_within = (self._N - k_fe) / (self._N - self._k - k_fe)
        else:
            adj_factor = (self._N - has_intercept) / (self._N - self._k)

        ssu = np.sum(self._u_hat**2)
        ssy = np.sum(self._weights * (Y - np.average(Y, weights=self._weights)) ** 2)
        self._rmse = np.sqrt(ssu / self._N)
        self._r2 = 1 - (ssu / ssy)
        self._adj_r2 = 1 - (ssu / ssy) * adj_factor

        if self._has_fixef:
            ssy_within = np.sum(Y_within**2)
            self._r2_within = 1 - (ssu / ssy_within)
            self._adj_r2_within = 1 - (ssu / ssy_within) * adj_factor_within

    def tidy(
        self,
        alpha: float = 0.05,
        inference_type: InferenceType = "regular",
    ) -> pd.DataFrame:
        """
        Tidy model outputs.

        Return a tidy pd.DataFrame with the point estimates, standard errors,
        t-statistics, and p-values.

        Parameters
        ----------
        alpha: Optional[float]
            The significance level for the confidence intervals. If None,
            computes a 95% confidence interval (`alpha = 0.05`).
        inference_type : {"regular"}, optional
            Type of coefficient-wise inference to report. Only `"regular"` is
            currently available. Defaults to `"regular"`.

        Returns
        -------
        tidy_df : pd.DataFrame
            A tidy pd.DataFrame containing the regression results, including point
            estimates, standard errors, t-statistics, and p-values.
        """
        inference_type = self._normalize_inference_type(inference_type)
        if inference_type == "simult":
            raise ValueError(
                "tidy() does not support inference_type='simult'. Use "
                "confint(inference_type='simult') for simultaneous intervals."
            )
        if inference_type == "savi":
            raise NotImplementedError(
                "inference_type='savi' is not available in tidy() yet."
            )

        ub, lb = 1 - alpha / 2, alpha / 2
        try:
            self.get_inference(alpha=alpha)
        except EmptyVcovError:
            warnings.warn(
                "Empty variance-covariance matrix detected",
                UserWarning,
            )

        data = {
            "Coefficient": self._coefnames,
            "Estimate": self._beta_hat,
            "Std. Error": self._se,
            "t value": self._tstat,
            "Pr(>|t|)": self._pvalue,
            # use slice because self._conf_int might be empty
            f"{lb * 100:.1f}%": self._conf_int[:1].flatten(),
            f"{ub * 100:.1f}%": self._conf_int[1:2].flatten(),
        }
        if (
            getattr(self, "_sample_split_var", None) is not None
            and (sample := getattr(self, "_sample_split_value", None)) is not None
        ):
            data["Sample"] = sample
        return pd.DataFrame(data).set_index("Coefficient")

    def _normalize_inference_type(
        self, inference_type: InferenceType, joint: bool = False
    ) -> InferenceType:
        """Validate `inference_type` and fold the deprecated `joint` flag into it."""
        _validate_literal_argument(inference_type, InferenceType)

        if joint:
            warnings.warn(
                "joint=True is deprecated. Use inference_type='simult' instead.",
                FutureWarning,
                stacklevel=3,
            )
            if inference_type not in ("regular", "simult"):
                raise ValueError(
                    "joint=True cannot be combined with "
                    f"inference_type={inference_type!r}."
                )
            inference_type = "simult"

        return inference_type

    def confint(
        self,
        alpha: float = 0.05,
        keep: list | str | None = None,
        drop: list | str | None = None,
        exact_match: bool | None = False,
        joint: bool = False,
        seed: int | None = None,
        reps: int = 10_000,
        *,
        inference_type: InferenceType = "regular",
        mixture_precision: float = 1.0,
    ) -> pd.DataFrame:
        r"""
        Fitted model confidence intervals.

        Parameters
        ----------
        alpha : float, optional
            The significance level for confidence intervals. Defaults to 0.05.
            keep: str or list of str, optional
        joint : bool, optional
            Deprecated. Use `inference_type="simult"` instead. Whether to
            compute simultaneous confidence intervals for the joint null of the
            parameters selected by `keep` and `drop`. Defaults to False. See
            https://www.causalml-book.org/assets/chapters/CausalML_chap_4.pdf,
            Remark 4.4.1 for details.
        keep: str or list of str, optional
            The pattern for retaining coefficient names. You can pass a string (one
            pattern) or a list (multiple patterns). Default is keeping all coefficients.
            You should use regular expressions to select coefficients.
                "age",            # would keep all coefficients containing age
                r"^tr",           # would keep all coefficients starting with tr
                r"\\d$",          # would keep all coefficients ending with number
            Output will be in the order of the patterns.
        drop: str or list of str, optional
            The pattern for excluding coefficient names. You can pass a string (one
            pattern) or a list (multiple patterns). Syntax is the same as for `keep`.
            Default is keeping all coefficients. Parameter `keep` and `drop` can be
            used simultaneously.
        exact_match: bool, optional
            Whether to use exact match for `keep` and `drop`. Default is False.
            If True, the pattern will be matched exactly to the coefficient name
            instead of using regular expressions.
        reps : int, optional
            The number of bootstrap iterations to run for joint confidence intervals.
            Defaults to 10_000. Only used if `joint` is True.
        seed : int, optional
            The seed for the random number generator. Defaults to None. Only used
            when `inference_type="simult"`.
        inference_type : {"regular", "simult", "savi"}, optional
            Type of confidence interval to compute. "regular" returns pointwise
            intervals; "simult" returns simultaneous (joint) intervals for the
            coefficients selected by `keep` and `drop`; "savi" returns
            coefficient-wise asymptotic SAVI confidence sequences. Defaults to
            "regular". Supersedes the deprecated `joint` argument. Keyword-only.
        mixture_precision: float, optional
            When `inference_type="savi"`, controls the mixing weight of the
            prior in the SAVI e-value. Larger values produce wider confidence
            sequences early on but narrow faster as the sample grows. Must
            remain fixed during sequential monitoring. Defaults to 1. This
            argument is keyword-only. Use
            `pyfixest.estimation.post_estimation.savi.optimal_mixture_precision()`
            to minimize confidence-sequence width at a target sample size.

        Returns
        -------
        pd.DataFrame
            A pd.DataFrame with confidence intervals of the estimated regression model
            for the selected coefficients.

        Notes
        -----
        SAVI currently supports unweighted, non-IV `feols` models without
        absorbed fixed effects. The covariance estimator must be iid or
        heteroskedasticity robust (`hetero`, `HC1`, `HC2`, or `HC3`). Direct
        `FixestMulti.confint()` calls provide regular inference only.

        Examples
        --------
        ```{python}
        #| echo: true
        #| results: asis
        #| include: true

        from pyfixest.utils import get_data
        from pyfixest.estimation import feols

        data = get_data()
        fit = feols("Y ~ C(f1)", data=data)
        fit.confint(alpha=0.10).head()
        fit.confint(alpha=0.10, inference_type="simult", reps=9999).head()

        savi_fit = feols("Y ~ X1 + X2", data=data, vcov="hetero")
        savi_fit.confint(alpha=0.10, inference_type="savi").head()
        ```
        """
        inference_type = self._normalize_inference_type(inference_type, joint=joint)
        if inference_type == "savi":
            from pyfixest.estimation.post_estimation.savi import _confint

            return _confint(
                model=self,
                alpha=alpha,
                mixture_precision=mixture_precision,
                keep=keep,
                drop=drop,
                exact_match=exact_match,
            )

        coefnames, coef_indices = _select_coefnames_and_indices(
            self._coefnames, keep, drop, exact_match
        )

        if inference_type == "regular":
            crit_val = self._inference_dist.crit_val(alpha, self._df_t)
        else:
            joint_indices = sorted(coef_indices)
            D_inv = 1 / self._se[joint_indices]
            V = self._vcov[np.ix_(joint_indices, joint_indices)]
            C_coefs = (D_inv * V).T * D_inv
            crit_val = simultaneous_crit_val(C_coefs, reps, alpha=alpha, seed=seed)

        ub = pd.Series(self._beta_hat[coef_indices] + crit_val * self._se[coef_indices])
        lb = pd.Series(self._beta_hat[coef_indices] - crit_val * self._se[coef_indices])

        df = pd.DataFrame(
            {
                f"{alpha / 2 * 100:.1f}%": lb,
                f"{(1 - alpha / 2) * 100:.1f}%": ub,
            }
        )
        # df = pd.DataFrame({f"{alpha / 2}%": lb, f"{1-alpha / 2}%": ub})
        df.index = coefnames

        return df

    def resid(self) -> np.ndarray:
        """
        Fitted model residuals.

        Returns
        -------
        np.ndarray
            A np.ndarray with the residuals of the estimated regression model.
        """
        return self._u_hat.flatten() / np.sqrt(self._weights.flatten())
