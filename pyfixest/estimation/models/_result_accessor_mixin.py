import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyfixest.errors import EmptyVcovError
from pyfixest.estimation.internals.model_state import CoefficientTable
from pyfixest.estimation.internals.retention import require_retained

if TYPE_CHECKING:
    from pyfixest.estimation.formula.model_matrix import ModelMatrix
    from pyfixest.estimation.internals.fit_statistics import FitStatistics
    from pyfixest.estimation.internals.model_state import (
        EstimationOptions,
        EstimationSample,
        ModelDescription,
        ObservationWeights,
        VarianceCovariance,
        WithinLinearData,
    )
    from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.internals.literals import (
    InferenceType,
    _validate_literal_argument,
)
from pyfixest.utils.dev_utils import _find_stack_level, _select_coefnames_and_indices
from pyfixest.utils.utils import simultaneous_crit_val


class TidyColumnAccessors:
    """Mixin: derive `coef/se/tstat/pvalue` from `tidy()` data frame."""

    def tidy(self, *args, **kwargs) -> pd.DataFrame:
        """Tidy DataFrame of results. Implemented by the host class."""
        raise NotImplementedError

    def coef(self) -> pd.Series:
        """
        Estimated coefficients as a pandas `Series`.

        Access as `fit.coef()` on a supported fitted result.
        Returns the `Estimate` column of `tidy()`.

        Returns
        -------
        pandas.Series
            Point estimates, indexed by coefficient name. On a
            `FixestMulti`, the index has the levels `fml` and `Coefficient`.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.coef()
        ```

        See Also
        --------
        [fit.tidy()](/reference/estimation.models.feols_.Feols.tidy.qmd) returns the full coefficient table;
        [fit.confint()](/reference/estimation.models.feols_.Feols.confint.qmd) returns confidence intervals.
        See [fitted results](/reference-guide/fitted-results.qmd) for other result accessors.
        """
        return self.tidy()["Estimate"]

    def se(self) -> pd.Series:
        """
        Coefficient standard errors as a pandas `Series`.

        Access as `fit.se()` on a supported fitted result.
        Returns the `Std. Error` column of `tidy()`. The values depend on the
        variance estimator of the model, which can be changed via `vcov()`.

        Returns
        -------
        pandas.Series
            Standard errors, indexed by coefficient name. On a
            `FixestMulti`, the index has the levels `fml` and `Coefficient`.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.se()
        ```

        See Also
        --------
        [fit.tidy()](/reference/estimation.models.feols_.Feols.tidy.qmd) returns the full coefficient table;
        [fit.confint()](/reference/estimation.models.feols_.Feols.confint.qmd) returns confidence intervals.
        See [fitted results](/reference-guide/fitted-results.qmd) for other result accessors.

        Notes
        -----
        Values use the model's current covariance estimator. Calling
        `fit.vcov(...)` changes subsequent inference; it updates the fitted
        result in place. These coefficient summaries remain available with
        `store_data=False` or `lean=True`.
        """
        return self.tidy()["Std. Error"]

    def tstat(self) -> pd.Series:
        """
        Coefficient t-statistics as a pandas `Series`.

        Access as `fit.tstat()` on a supported fitted result.
        Returns the `t value` column of `tidy()`, i.e. each estimate divided by
        its standard error.

        Returns
        -------
        pandas.Series
            t-statistics, indexed by coefficient name. On a
            `FixestMulti`, the index has the levels `fml` and `Coefficient`.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.tstat()
        ```

        See Also
        --------
        [fit.tidy()](/reference/estimation.models.feols_.Feols.tidy.qmd) returns the full coefficient table;
        [fit.confint()](/reference/estimation.models.feols_.Feols.confint.qmd) returns confidence intervals.
        See [fitted results](/reference-guide/fitted-results.qmd) for other result accessors.

        Notes
        -----
        Values use the model's current covariance estimator. Calling
        `fit.vcov(...)` changes subsequent inference; it updates the fitted
        result in place. These coefficient summaries remain available with
        `store_data=False` or `lean=True`.
        """
        return self.tidy()["t value"]

    def pvalue(self) -> pd.Series:
        """
        Coefficient p-values as a pandas `Series`.

        Access as `fit.pvalue()` on a supported fitted result.
        Returns the `Pr(>|t|)` column of `tidy()`, for the two-sided null that a
        coefficient is zero.

        Returns
        -------
        pandas.Series
            p-values, indexed by coefficient name. On a
            `FixestMulti`, the index has the levels `fml` and `Coefficient`.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.pvalue()
        ```

        See Also
        --------
        [fit.tidy()](/reference/estimation.models.feols_.Feols.tidy.qmd) returns the full coefficient table;
        [fit.confint()](/reference/estimation.models.feols_.Feols.confint.qmd) returns confidence intervals.
        See [fitted results](/reference-guide/fitted-results.qmd) for other result accessors.

        Notes
        -----
        Values use the model's current covariance estimator. Calling
        `fit.vcov(...)` changes subsequent inference; it updates the fitted
        result in place. These coefficient summaries remain available with
        `store_data=False` or `lean=True`.
        """
        return self.tidy()["Pr(>|t|)"]


class ResultAccessorMixin(TidyColumnAccessors):
    """Mixin providing result-accessor methods for fitted models."""

    # Type declarations for attributes provided by the host class (Feols).
    variance_covariance: "VarianceCovariance"
    coeftable: CoefficientTable
    _beta_hat: np.ndarray
    _u_hat: np.ndarray
    model_matrix: "ModelMatrix"
    options: "EstimationOptions"
    observation_weights: "ObservationWeights"
    sample_info: "EstimationSample"
    within_data: "WithinLinearData"
    fitstat: "FitStatistics"
    _coefnames: list[str]
    model: "ModelDescription"
    _k: int

    @property
    def _fml(self) -> str:
        """Formula string. Kept for third-party integrations (e.g. marginaleffects's `ModelPyfixest`) that read this private attribute directly; use `model.formula` instead."""
        return self.model.formula

    @property
    def _method(self) -> str:
        """Estimator name. Kept for third-party integrations (e.g. marginaleffects's `ModelPyfixest`) that read this private attribute directly; use `model.method` instead."""
        return self.model.method

    @property
    def _vcov(self) -> np.ndarray:
        """Covariance matrix. Kept for third-party integrations (e.g. marginaleffects's `ModelPyfixest`) that read this private attribute directly; use `variance_covariance.vcov` instead."""
        return self.variance_covariance.vcov

    def evalue(
        self: "Feols",
        mixture_precision: float = 1.0,
    ) -> pd.Series:
        """
        Compute coefficient-wise asymptotic SAVI e-values.

        Access as `fit.evalue(...)` on a supported fitted result.

        Parameters
        ----------
        mixture_precision : float, optional
            Positive mixture precision fixed before sequential monitoring.
            Defaults to 1. Use `pf.optimal_mixture_precision()` to
            minimize confidence-sequence width at a target sample size.

        Returns
        -------
        pd.Series
            One e-value for each two-sided null that a coefficient equals zero,
            indexed by coefficient name, with `Series` name `"e_value"`.

        Notes
        -----
        SAVI currently supports unweighted, non-IV `feols` models without
        absorbed fixed effects. The covariance estimator must be iid or
        heteroskedasticity robust (`hetero`, `HC1`, `HC2`, or `HC3`). Note that
        for `HC2`/`HC3`, pyfixest's default small-sample correction scales the
        variance by `n / (n - k)` while the R implementation in `avlm` does not.
        Inference is coefficient-wise, without multiplicity adjustment. Fix
            `mixture_precision` before monitoring. An e-value at least `1 / alpha`
            is evidence against the corresponding null at level `alpha`; this uses
            the asymptotic SAVI construction, not a finite-sample guarantee.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2", data=data, vcov="hetero")
        fit.evalue()
        ```

        See Also
        --------
        [fit.pvalue_savi()](/reference/estimation.models.feols_.Feols.pvalue_savi.qmd) reports `min(1, 1 / evalue)`.
        [fit.confint()](/reference/estimation.models.feols_.Feols.confint.qmd) supports `inference_type="savi"`.
        [pf.optimal_mixture_precision()](/reference/estimation.post_estimation.savi.optimal_mixture_precision.qmd)
        connects planning, tuning and inference in a working example.
        """
        from pyfixest.estimation.post_estimation.savi import (
            _evalue,
            _validate_savi_model,
        )

        _validate_savi_model(model=self, method="evalue")
        return _evalue(model=self, mixture_precision=mixture_precision)

    def pvalue_savi(
        self: "Feols",
        mixture_precision: float = 1.0,
    ) -> pd.Series:
        """
        Compute coefficient-wise asymptotic SAVI sequential p-values.

        Access as `fit.pvalue_savi(...)` on a supported fitted result.
        Returns `min(1, 1 / evalue)` for each two-sided zero-coefficient null.
        See [fit.evalue()](/reference/estimation.models.feols_.Feols.evalue.qmd)
        for the interpretation and supported-model restrictions.

        Parameters
        ----------
        mixture_precision : float, optional
            Positive mixture precision fixed before sequential monitoring.
            Defaults to 1. Use `pf.optimal_mixture_precision()` to
            minimize confidence-sequence width at a target sample size.

        Returns
        -------
        pd.Series
            Sequential p-values indexed by coefficient name, with `Series` name
            `"Pr(>|t|)"`. There is no adjustment across coefficients.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2", data=data, vcov="HC1")
        fit.pvalue_savi()
        ```

        Notes
        -----
        Supported only for unweighted, non-IV OLS without absorbed fixed
        effects and with iid, hetero, HC1, HC2 or HC3 covariance. Mixture
        precision must be fixed before monitoring. SAVI inference is asymptotic;
        see `fit.evalue()` for the small-sample scaling convention.

        See Also
        --------
        [fit.evalue()](/reference/estimation.models.feols_.Feols.evalue.qmd) returns the underlying e-values;
        [fit.pvalue()](/reference/estimation.models.feols_.Feols.pvalue.qmd) returns ordinary fixed-sample p-values.
        [pf.optimal_mixture_precision()](/reference/estimation.post_estimation.savi.optimal_mixture_precision.qmd)
        shows tuning at a planned sample size.
        """
        from pyfixest.estimation.post_estimation.savi import (
            _pvalue_savi,
            _validate_savi_model,
        )

        _validate_savi_model(model=self, method="pvalue_savi")
        return _pvalue_savi(model=self, mixture_precision=mixture_precision)

    def get_inference(self, alpha: float = 0.05) -> None:
        """
        Publish `coeftable`, the coefficient table of the current covariance.

        Parameters
        ----------
        alpha : float, optional
            The significance level for confidence intervals. Defaults to 0.05, which
            produces a 95% confidence interval.

        Returns
        -------
        None
            The table is stored as `coeftable`, a
            [CoefficientTable](/reference/estimation.state.CoefficientTable.qmd).

        Details
        -------
        relevant fixest functions:
        - fixest_CI_factor: https://github.com/lrberge/fixest/blob/5523d48ef4a430fa2e82815ca589fc8a47168fe7/R/miscfuns.R#L5614
        """
        if not hasattr(self, "variance_covariance"):
            raise EmptyVcovError()
        covariance = self.variance_covariance
        dist = self.model.inference_dist

        beta_hat = self._beta_hat
        se = np.sqrt(np.diagonal(covariance.vcov))
        tstat = beta_hat / se
        pvalue = dist.pvalue(tstat, covariance.df_t)
        # fixest_CI_factor: beta +- q(1 - alpha / 2) * se at df_t degrees of freedom
        z_se = dist.crit_val(alpha, covariance.df_t) * se
        self.coeftable = CoefficientTable(
            estimate=beta_hat,
            se=se,
            tstat=tstat,
            pvalue=pvalue,
            conf_int=np.array([beta_hat - z_se, beta_hat + z_se]),
            alpha=alpha,
        )

    def tidy(
        self,
        alpha: float = 0.05,
        inference_type: InferenceType = "regular",
    ) -> pd.DataFrame:
        """
        Tidy model outputs.

        Access as `fit.tidy(...)` on a supported fitted result.
        Return a pandas `DataFrame` with the point estimates, standard errors,
        t-statistics, p-values and confidence intervals.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence intervals. Defaults to
            0.05, which gives a 95% confidence interval.
        inference_type : {"regular"}, optional
            Type of coefficient-wise inference to report. Only `"regular"` is
            currently available. Defaults to `"regular"`.

        Returns
        -------
        tidy_df : pd.DataFrame
            A data frame indexed by `Coefficient`, with the columns `Estimate`,
            `Std. Error`, `t value`, `Pr(>|t|)` and the lower and upper
            confidence bounds (e.g. `2.5%` and `97.5%` for `alpha=0.05`). Fits
            estimated with `split` or `fsplit` add a `Sample` column.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.tidy()
        ```

        Changing the variance estimator changes the standard errors, t-values
        and p-values reported by `tidy()`.

        ```{python}
        fit.vcov("hetero").tidy()
        ```

        Notes
        -----
        Uses the current covariance estimator and refreshes `fit.coeftable`
        with the requested `alpha`. Coefficient summaries remain available with
        `store_data=False` or `lean=True`. For simultaneous intervals or SAVI
        confidence sequences, use `fit.confint(inference_type=...)` instead.

        See Also
        --------
        [fit.coef()](/reference/estimation.models.feols_.Feols.coef.qmd) and
        [fit.se()](/reference/estimation.models.feols_.Feols.se.qmd) extract individual columns.
        [fit.vcov()](/reference/estimation.models.feols_.Feols.vcov.qmd) changes inference;
        [pf.etable()](/reference/report.etable.qmd) compares fitted models.
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
            table = self.coeftable
            se, tstat, pvalue = table.se, table.tstat, table.pvalue
            conf_int = table.conf_int
        except EmptyVcovError:
            warnings.warn(
                "Empty variance-covariance matrix detected",
                UserWarning,
                stacklevel=_find_stack_level(),
            )
            # Fixed-effects-only model: no coefficients, so no inference rows.
            se = tstat = pvalue = np.empty(0)
            conf_int = np.empty((2, 0))

        data = {
            "Coefficient": self._coefnames,
            "Estimate": self._beta_hat,
            "Std. Error": se,
            "t value": tstat,
            "Pr(>|t|)": pvalue,
            f"{lb * 100:.1f}%": conf_int[0],
            f"{ub * 100:.1f}%": conf_int[1],
        }
        if self.model.sample_split is not None:
            sample = self.model.sample_split.value
            data["Sample"] = "all" if sample is None else sample
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
        self: "Feols",
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
        Compute pointwise, simultaneous or SAVI confidence intervals.

        Access as `fit.confint(...)` on a supported fitted result.

        Parameters
        ----------
        alpha : float, optional
            The significance level for confidence intervals. Defaults to 0.05.
        joint : bool, optional
            Deprecated. Use `inference_type="simult"` instead. Whether to
            compute simultaneous confidence intervals for the joint null of the
            parameters selected by `keep` and `drop`. Defaults to `False`. See
            https://www.causalml-book.org/assets/chapters/CausalML_chap_4.pdf,
            Remark 4.4.1 for details.
        keep : str or list of str, optional
            The pattern for retaining coefficient names. You can pass a string (one
            pattern) or a list (multiple patterns). Default is keeping all coefficients.
            Patterns are regular expressions; see the Notes for examples.
            Output will be in the order of the patterns.
        drop : str or list of str, optional
            The pattern for excluding coefficient names. You can pass a string (one
            pattern) or a list (multiple patterns). Syntax is the same as for `keep`.
            Default is keeping all coefficients. Parameter `keep` and `drop` can be
            used simultaneously.
        exact_match : bool, optional
            Whether to use exact match for `keep` and `drop`. Default is `False`.
            If `True`, the pattern will be matched exactly to the coefficient name
            instead of using regular expressions.
        reps : int, optional
            The number of simulation draws for the simultaneous confidence
            intervals. Defaults to 10_000. Only used when
            `inference_type="simult"`.
        seed : int, optional
            The seed for the random number generator. Defaults to `None`. Only used
            when `inference_type="simult"`.
        inference_type : {"regular", "simult", "savi"}, optional
            Type of confidence interval to compute. `regular` returns pointwise
            intervals; `simult` returns simultaneous (joint) intervals for the
            coefficients selected by `keep` and `drop`; `savi` returns
            coefficient-wise asymptotic SAVI confidence sequences. Defaults to
            `regular`. Supersedes the deprecated `joint` argument.
        mixture_precision : float, optional
            Positive mixture precision for `inference_type="savi"`, fixed before
            monitoring begins. Defaults to 1. Choose it with
            `pf.optimal_mixture_precision()` at a planned sample size;
            do not retune it after inspecting the outcomes.

        Returns
        -------
        pd.DataFrame
            Lower and upper bounds, indexed by selected coefficient name.
            Columns are labeled by the tail probabilities, e.g. `"2.5%"` and
            `"97.5%"` at `alpha=0.05`. Selected coefficients follow `keep`
            pattern order; `drop` then excludes matches. Simultaneous coverage
            is for the selected family, while SAVI is coefficient-wise.

        Notes
        -----
        SAVI currently supports unweighted, non-IV `feols` models without
        absorbed fixed effects. The covariance estimator must be iid or
        heteroskedasticity robust (`hetero`, `HC1`, `HC2`, or `HC3`). With
        `HC2`/`HC3`, pyfixest's default small-sample correction scales the
        variance by `n / (n - k)`. You need to pass `ssc(k_adj=False)` to reproduce `avlm`,
        which applies no such correction.

        `keep` and `drop` take regular expressions, for example:

        ```python
        "age"  # keeps all coefficients containing "age"

        r"^tr"  # keeps all coefficients starting with "tr"
        r"\d$"  # keeps all coefficients ending with a digit
        ```

        Examples
        --------
        ```{python}
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2", data=data, vcov="hetero")
        fit.confint(alpha=0.10)
        ```

        Simultaneous intervals for the two slopes:

        ```{python}
        fit.confint(
            keep=["X1", "X2"], exact_match=True, inference_type="simult",
            reps=999, seed=123,
        )
        ```

        Asymptotic SAVI confidence sequences with a fixed mixture precision:

        ```{python}
        fit.confint(inference_type="savi", mixture_precision=1.0)
        ```

        See Also
        --------
        [fit.tidy()](/reference/estimation.models.feols_.Feols.tidy.qmd) returns regular inference in one table.
        [fit.evalue()](/reference/estimation.models.feols_.Feols.evalue.qmd) and
        [fit.pvalue_savi()](/reference/estimation.models.feols_.Feols.pvalue_savi.qmd)
        report SAVI tests. [pf.optimal_mixture_precision()](/reference/estimation.post_estimation.savi.optimal_mixture_precision.qmd)
        shows how to tune the confidence-sequence width before monitoring.
        """
        inference_type = self._normalize_inference_type(inference_type, joint=joint)
        if inference_type == "savi":
            from pyfixest.estimation.post_estimation.savi import (
                _confint,
                _validate_savi_model,
            )

            _validate_savi_model(model=self, method="confint")
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

        se = self.coeftable.se
        if inference_type == "regular":
            crit_val = self.model.inference_dist.crit_val(
                alpha, self.variance_covariance.df_t
            )
        else:
            joint_indices = sorted(coef_indices)
            D_inv = 1 / se[joint_indices]
            V = self.variance_covariance.vcov[np.ix_(joint_indices, joint_indices)]
            C_coefs = (D_inv * V).T * D_inv
            crit_val = simultaneous_crit_val(C_coefs, reps, alpha=alpha, seed=seed)

        ub = pd.Series(self._beta_hat[coef_indices] + crit_val * se[coef_indices])
        lb = pd.Series(self._beta_hat[coef_indices] - crit_val * se[coef_indices])

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

        Access as `fit.resid()` on a supported fitted result.
        Residuals are stored and returned on the scale of the original dependent
        variable. Observation weights are applied only where an estimating
        equation or diagnostic requires them. GLM fits override this method
        with a `type` argument (`"response"` or `"working"` residuals); see
        [Feglm.resid()](/reference/estimation.models.feglm_.Feglm.qmd).

        Returns
        -------
        np.ndarray
            A one-dimensional array of length `fit.sample_info.n_rows`, in
            estimation-sample order after missing-row and singleton removal
            (and separation removal for GLMs). It is not padded to the original
            input length. Observation weights do not rescale these residuals.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
        fit.resid()[:5]
        ```

        Notes
        -----
        Residuals require retained arrays and are unavailable with `lean=True`.
        `store_data=False` alone retains them. GLMs provide their own `type`
        argument to choose response or working residuals; the default is
        response residuals.

        See Also
        --------
        [fit.predict()](/reference/estimation.models.feols_.Feols.predict.qmd) returns fitted or new-data predictions.
        [EstimationSample](/reference/estimation.state.EstimationSample.qmd) records sample alignment.
        See [model storage](/reference-guide/model-storage.qmd) for retained-state requirements.
        """
        require_retained(self, "resid", "_u_hat")
        return self._u_hat.flatten()
