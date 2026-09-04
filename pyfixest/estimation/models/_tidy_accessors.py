"""Column accessors shared by fitted results and the multi-model container."""

import pandas as pd


class TidyColumnAccessors:
    """Mixin: derive `coef/se/tstat/pvalue` from `tidy()` data frame."""

    def tidy(self, *args, **kwargs) -> pd.DataFrame:
        """Tidy DataFrame of results. Implemented by the host class."""
        raise NotImplementedError

    def coef(self) -> pd.Series:
        """
        Estimated coefficients as a pandas Series.

        Returns the `Estimate` column of `tidy()`.

        Returns
        -------
        pandas.Series
            Point estimates, indexed by coefficient name.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.coef()
        ```
        """
        return self.tidy()["Estimate"]

    def se(self) -> pd.Series:
        """
        Coefficient standard errors as a pandas Series.

        Returns the `Std. Error` column of `tidy()`. The values depend on the
        variance estimator of the model, which can be changed via `vcov()`.

        Returns
        -------
        pandas.Series
            Standard errors, indexed by coefficient name.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.se()
        ```
        """
        return self.tidy()["Std. Error"]

    def tstat(self) -> pd.Series:
        """
        Coefficient t-statistics as a pandas Series.

        Returns the `t value` column of `tidy()`, i.e. each estimate divided by
        its standard error.

        Returns
        -------
        pandas.Series
            t-statistics, indexed by coefficient name.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.tstat()
        ```
        """
        return self.tidy()["t value"]

    def pvalue(self) -> pd.Series:
        """
        Coefficient p-values as a pandas Series.

        Returns the `Pr(>|t|)` column of `tidy()`, for the two-sided null that a
        coefficient is zero.

        Returns
        -------
        pandas.Series
            p-values, indexed by coefficient name.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.pvalue()
        ```
        """
        return self.tidy()["Pr(>|t|)"]
