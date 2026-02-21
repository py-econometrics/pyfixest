import functools
import warnings
from importlib import import_module
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from pyfixest.errors import EmptyVcovError
from pyfixest.utils.dev_utils import _select_order_coefs
from pyfixest.utils.utils import simultaneous_crit_val


class ResultAccessorMixin:
    """Mixin providing result-accessor methods for fitted models."""

    def _bind_report_methods(self):
        """Bind summary, coefplot, and iplot from pyfixest.report as instance methods."""
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
        # use t-dist for linear models, but normal for non-linear models
        if self._method in ["fepois", "feglm-probit", "feglm-logit", "feglm-gaussian"]:
            self._pvalue = 2 * (1 - norm.cdf(np.abs(self._tstat)))
            z = np.abs(norm.ppf(alpha / 2))
        else:
            self._pvalue = 2 * (1 - t.cdf(np.abs(self._tstat), self._df_t))
            z = np.abs(t.ppf(alpha / 2, self._df_t))

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

        Returns
        -------
        tidy_df : pd.DataFrame
            A tidy pd.DataFrame containing the regression results, including point
            estimates, standard errors, t-statistics, and p-values.
        """
        ub, lb = 1 - alpha / 2, alpha / 2
        try:
            self.get_inference(alpha=alpha)
        except EmptyVcovError:
            warnings.warn(
                "Empty variance-covariance matrix detected",
                UserWarning,
            )

        tidy_df = pd.DataFrame(
            {
                "Coefficient": self._coefnames,
                "Estimate": self._beta_hat,
                "Std. Error": self._se,
                "t value": self._tstat,
                "Pr(>|t|)": self._pvalue,
                # use slice because self._conf_int might be empty
                f"{lb * 100:.1f}%": self._conf_int[:1].flatten(),
                f"{ub * 100:.1f}%": self._conf_int[1:2].flatten(),
            }
        )

        return tidy_df.set_index("Coefficient")

    def coef(self) -> pd.Series:
        """
        Fitted model coefficents.

        Returns
        -------
        pd.Series
            A pd.Series with the estimated coefficients of the regression model.
        """
        return self.tidy()["Estimate"]

    def se(self) -> pd.Series:
        """
        Fitted model standard errors.

        Returns
        -------
        pd.Series
            A pd.Series with the standard errors of the estimated regression model.
        """
        return self.tidy()["Std. Error"]

    def tstat(self) -> pd.Series:
        """
        Fitted model t-statistics.

        Returns
        -------
        pd.Series
            A pd.Series with t-statistics of the estimated regression model.
        """
        return self.tidy()["t value"]

    def pvalue(self) -> pd.Series:
        """
        Fitted model p-values.

        Returns
        -------
        pd.Series
            A pd.Series with p-values of the estimated regression model.
        """
        return self.tidy()["Pr(>|t|)"]

    def confint(
        self,
        alpha: float = 0.05,
        keep: Optional[Union[list, str]] = None,
        drop: Optional[Union[list, str]] = None,
        exact_match: Optional[bool] = False,
        joint: bool = False,
        seed: Optional[int] = None,
        reps: int = 10_000,
    ) -> pd.DataFrame:
        r"""
        Fitted model confidence intervals.

        Parameters
        ----------
        alpha : float, optional
            The significance level for confidence intervals. Defaults to 0.05.
            keep: str or list of str, optional
        joint : bool, optional
            Whether to compute simultaneous confidence interval for joint null
            of parameters selected by `keep` and `drop`. Defaults to False. See
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
            The seed for the random number generator. Defaults to None. Only used if
            `joint` is True.

        Returns
        -------
        pd.DataFrame
            A pd.DataFrame with confidence intervals of the estimated regression model
            for the selected coefficients.

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
        fit.confint(alpha=0.10, joint=True, reps=9999).head()
        ```
        """
        if keep is None:
            keep = []
        if drop is None:
            drop = []

        tidy_df = self.tidy()
        if keep or drop:
            if isinstance(keep, str):
                keep = [keep]
            if isinstance(drop, str):
                drop = [drop]
            idxs = _select_order_coefs(tidy_df.index.tolist(), keep, drop, exact_match)
            coefnames = tidy_df.loc[idxs, :].index.tolist()
        else:
            coefnames = self._coefnames

        joint_indices = [i for i, x in enumerate(self._coefnames) if x in coefnames]
        if not joint_indices:
            raise ValueError("No coefficients match the keep/drop patterns.")

        if not joint:
            if self._method == "feols":
                crit_val = np.abs(t.ppf(alpha / 2, self._df_t))
            else:
                crit_val = np.abs(norm.ppf(alpha / 2))
        else:
            D_inv = 1 / self._se[joint_indices]
            V = self._vcov[np.ix_(joint_indices, joint_indices)]
            C_coefs = (D_inv * V).T * D_inv
            crit_val = simultaneous_crit_val(C_coefs, reps, alpha=alpha, seed=seed)

        ub = pd.Series(
            self._beta_hat[joint_indices] + crit_val * self._se[joint_indices]
        )
        lb = pd.Series(
            self._beta_hat[joint_indices] - crit_val * self._se[joint_indices]
        )

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
