import sys
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import t

@dataclass
class SensitivityAnalysis:
    """
    Implements the sensitivity analysis method described in Cinelli and Hazlett (2020): "Making Sense of Sensitivity: Extending Omitted Variable Bias".

    This class performs the analysis, creates the benchmarks and supports visualizations and output creation.

    Parameters
    ----------
    model: pyfixest.Feols
        A fitted `pyfixest` model object (e.g., from `feols()`).
    X: str, Optional
        The name of the treatment variable to analyze by default. If None, operations requiring a treatment variable must specify it explicitly.
    """
    model: Any
    X: Optional[str] = None

   # let's start with R_2
    def partial_r2(self, X: Optional[str] = None) -> Union[float, np.ndarray]:
       """
       Calculate the partial R2 for a given variable.

       The partial R2 explains how much of the residual variance of the outcome is explained by the covariate.

       Parameters
       ----------
       X: str, Optional
            The name of the covariate for which to compute the partial R2. If None, returns partial R2s for all covariates in the model.

       Returns
       -------
       float or np.ndarray
            The partial R2 value(s).
       """
       df = self.model._df_t
       names = self.model._coefnames
       tstat = self.model._tstat

       if X is None:
           return tstat**2 / (tstat**2 + df)

       idx = names.index(X)
       return tstat[idx]**2 / (tstat[idx]**2 + df)

    # define partial f2
    def partial_f2(self, X: Optional[str] = None) -> Union[float, np.ndarray]:
        """
        Compute the partial (Cohen's) f2 for a linear regression model.

        The partial f2 is a measure of effect size (a transformation of the partial R2).

        Parameters
        ----------
        X: str, Optional
            The name of the covariate for which to compute the partial R2. If None, returns partial R2s for all covariates in the model.

        Returns
        -------
        float or np.ndarray
            The partial f2 value(s).
        """
        df = self.model._df_t
        names = self.model._coefnames
        tstat = self.model._tstat

        if X is None:
            return tstat**2 / df

        idx = names.index(X)
        return tstat[idx]**2 / df

    # robustness value function
    def robustness_value(self, X: Optional[str] = None, q = 1, alpha = 1.0) -> Union[float, np.ndarray]:
        """
        Compute the robustness value (RV) of the regression coefficient.

        The RV describes the minimum strength of association (partial R2) that an unobserved confounder would need to have with both the treatment and the outcome to change the research conclusions (e.g., reduce effect by q% or render it insignificant).

        Parameters
        ----------
        X : str, optional
            The name of the covariate.
        q : float, default 1.0
            The proportion of reduction in the coefficient estimate (e.g., 1.0 = 100% reduction to zero).
        alpha : float, default 1.0
            The significance level. If 1.0 (default), computes RV for the point estimate (RV_q).
            If < 1.0 (e.g., 0.05), computes RV for statistical significance (RV_qa)

        Returns
        -------
        float or np.ndarray
            The robustness value(s).
        """
        df = self.model._df_t
        f2 = self.partial_f2(X = X)

        fq = q * np.sqrt(f2)
        f_crit = abs(t.ppf(alpha / 2, df - 1)) / np.sqrt(df - 1)
        fqa = fq - f_crit

        rv = np.where(fqa > 0, 0.5 * (np.sqrt(fqa ** 4 + 4 * fqa ** 2) - fqa ** 2), 0.0)

        # check edge cases
        edge_case = 1 - (1 / fq**2)
        rv = np.where(rv > edge_case, rv, (fq**2 - f_crit**2) / (1 + fq**2))

        return rv

    # sensitivity stats function to report these
    def sensitivity_stats(self, X: Optional[str] = None, q = 1, alpha = 0.05) -> dict:
        """
        Compute the sensitivity statistics for the model.

        Parameters
        ----------
        X : str, optional
            The name of the covariate.
        q : float, default 1.0
            The percent reduction for the Robustness Value.
        alpha : float, default 0.05
            The significance level for the Robustness Value.

        Returns
        -------
        dict
            A dictionary containing:
            - 'estimate': Coefficient estimate
            - 'se': Standard Error
            - 'df': Degrees of Freedom
            - 'partial_R2': Partial R2 of the covariate
            - 'partial_f2': Partial f2 of the covariate
            - 'rv_q': Robustness Value for point estimate
            - 'rv_qa': Robustness Value for statistical significance
        """
        estimate = self.model._beta_hat
        se = self.model._se
        df = self.model._df_t

        if X is not None:
            idx = self.model._coefnames.index(X)
            estimate = estimate[idx]
            se = se[idx]

        # compute statistics
        r2yd_x = self.partial_r2(X = X)
        f2yd_x = self.partial_f2(X = X)
        rv_q = self.robustness_value(X = X, q = q, alpha = 1) # alpha = 1 makes f_crit = 0
        rv_qa = self.robustness_value(X = X, q = q, alpha = alpha)

        sensitivity_stats_df = {'estimate': estimate, 'se': se, 'df': df, 'partial_R2': r2yd_x, 'partial_f2': f2yd_x, 'rv_q': rv_q, 'rv_qa': rv_qa }

        return sensitivity_stats_df

    # Compute Omitted Variable Bias Bounds
    def ovb_bounds(self, treatment, benchmark_covariates, kd=[1, 2, 3], ky=None, alpha=0.05, adjusted_estimate=True, bound="partial r2"):
        """
        Compute bounds on omitted variable bias using observed covariates as benchmarks.

        Parameters
        ----------
        treatment : str
            The name of the treatment variable.
        benchmark_covariates : str or list of str
            The names of the observed covariates to use for benchmarking.
        kd : float or list of floats, default [1, 2, 3]
            The multiplier for the strength of the confounder with the treatment
            relative to the benchmark covariate.
        ky : float or list of floats, optional
            The multiplier for the strength of the confounder with the outcome.
            If None, defaults to the same values as `kd`.
        alpha : float, default 0.05
            Significance level for computing confidence intervals of the adjusted estimates.
        adjusted_estimate : bool, default True
            If True, computes the adjusted estimate, SE, t-statistic, and CI.
        bound : str, default "partial r2"
            The type of bound to compute. Currently only "partial r2" is supported.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the sensitivity bounds and (optional) adjusted statistics.
        """
        df = self.model._df_t

        if ky is None:
            ky = kd

        if bound != "partial r2":
            raise ValueError("Only partial r2 is implemented as of now.")

        bounds = self._ovb_bounds_partial_r2(treatment = treatment, benchmark_covariates = benchmark_covariates, kd = kd, ky = ky)

        if adjusted_estimate:
            bounds['treatment'] = treatment
            bounds['adjusted_estimate'] = self.adjusted_estimate(bounds['r2dz_x'], bounds['r2yz_dx'], treatment = treatment, reduce = True)
            bounds['adjusted_se'] = self.adjusted_se(bounds['r2dz_x'], bounds['r2yz_dx'], treatment = treatment)
            bounds['adjusted_t'] = self.adjusted_t(bounds['r2dz_x'], bounds['r2yz_dx'], treatment = treatment, reduce = True, h0 = 0)
            se_multiple = abs(t.ppf(alpha / 2, df))
            bounds['adjusted_lower_CI'] = bounds['adjusted_estimate'] - se_multiple * bounds['adjusted_se']
            bounds['adjusted_upper_CI'] = bounds['adjusted_estimate'] + se_multiple * bounds['adjusted_se']

        return bounds

    def _ovb_bounds_partial_r2(self, treatment, benchmark_covariates, kd, ky):
        """
        Compute OVB bounds based on partial R2.

        This function should not be called directly. It is called under the ovb_bounds user facing function.

        Parameters
        ----------
        treatment : str
            The treatment variable.
        benchmark_covariates : str or list
            Benchmarks.
        kd, ky : float or list
            Strength multipliers.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'r2dz_x' and 'r2yz_dx' columns.
        """
        from pyfixest.estimation.api import feols
        model = self.model
        if (model is None or treatment is None):
            raise ValueError('ovb_partial_r2 requires a model object and a treatment variable')

        data = model._data
        X = pd.DataFrame(model._X, columns=model._coefnames)

        if treatment not in X.columns:
            raise ValueError(f"Treatment '{treatment}' not found in model coefficients.")

        non_treatment = X.drop(columns = treatment)
        covariate_names = [
            col for col in non_treatment.columns
            if col != "Intercept"
        ]
        covariates = ' + '.join(covariate_names)

        formula = f"{treatment} ~ {covariates}"

        treatment_model = feols(formula, data = data)
        treatment_sens = treatment_model.sensitivity_analysis()

        if isinstance(benchmark_covariates, str):
            benchmark_covariates = [benchmark_covariates]

        if np.isscalar(kd): kd = [kd]
        if np.isscalar(ky): ky = [ky]
        if len(ky) != len(kd):
            ky = ky * len(kd) if len(ky) == 1 else ky

        bounds_list = []

        for b in benchmark_covariates:
            r2yxj_dx = self.partial_r2(X = b)
            r2dxj_x = treatment_sens.partial_r2(X = b)

            for kd_val, ky_val in zip(kd, ky):
                r2dz_x = kd_val * (r2dxj_x / (1-r2dxj_x))

                if r2dz_x >= 1:
                    raise ValueError(f"Implied bound on r2dz.x >= 1 for benchmark {b} with kd={kd_val}."
                                     "Impossible scenario. Try a lower kd.")
                r2zxj_xd = kd_val * (r2dxj_x**2) / ((1 - kd_val * r2dxj_x) * (1 - r2dxj_x))

                if r2zxj_xd >= 1:
                    raise ValueError(f"Impossible kd value for benchmark {b}. Try a lower kd.")

                r2yz_dx = (((np.sqrt(ky_val) + np.sqrt(r2zxj_xd)) / np.sqrt(1 - r2zxj_xd))**2) * (r2yxj_dx / (1 - r2yxj_dx))

                if r2yz_dx > 1:
                    print(f"Warning: Implied bound on r2yz.dx > 1 for {b}. Capping at 1.")
                    r2yz_dx = 1.0

                bounds_list.append({
                    'bound_label': f"{kd_val}x {b}",  # Simple label maker
                    'r2dz_x': r2dz_x,
                    'r2yz_dx': r2yz_dx,
                    'benchmark_covariate': b,
                    'kd': kd_val,
                    'ky': ky_val
                })

        return pd.DataFrame(bounds_list)

    def bias(self, r2dz_x, r2yz_dx, treatment):
        """
        Compute the bias for the partial R2 parametrization.

        Parameters
        ----------
        r2dz_x : float or np.ndarray
            Partial R2 of confounder with treatment.
        r2yz_dx : float or np.ndarray
            Partial R2 of confounder with outcome.
        treatment : str
            The treatment variable.

        Returns
        -------
        float or np.ndarray
            The estimated bias amount (in units of the coefficient).
        """
        df = self.model._df_t
        idx = self.model._coefnames.index(treatment)
        se = self.model._se[idx]

        r2dz_x, r2yz_dx = np.array(r2dz_x), np.array(r2yz_dx)
        bias_factor = np.sqrt((r2yz_dx * r2dz_x) / (1 - r2dz_x))

        return bias_factor * se * np.sqrt(df)

    def adjusted_estimate(self, r2dz_x, r2yz_dx, treatment, reduce=True):
        """
        Compute the bias-adjusted coefficient estimate.

        Parameters
        ----------
        r2dz_x, r2yz_dx : float or np.ndarray
            Partial R2 parameters of the confounder.
        treatment : str
            The treatment variable.
        reduce : bool, default True
            If True, assumes bias moves the estimate toward zero (conservative).
            If False, assumes bias moves estimate away from zero.

        Returns
        -------
        float or np.ndarray
            The adjusted coefficient.
        """
        idx = self.model._coefnames.index(treatment)
        estimate = self.model._beta_hat[idx]

        if reduce:
            return np.sign(estimate) * (abs(estimate) - self.bias(r2dz_x, r2yz_dx, treatment = treatment))
        else:
            return np.sign(estimate) * (abs(estimate) + self.bias(r2dz_x, r2yz_dx, treatment = treatment))

    def adjusted_se(self, r2dz_x, r2yz_dx, treatment):
        """
        Compute the bias-adjusted Standard Error estimate.

        Parameters
        ----------
        r2dz_x, r2yz_dx : float or np.ndarray
            Partial R2 parameters of the confounder.
        treatment : str
            The treatment variable.

        Returns
        -------
        float or np.ndarray
            The adjusted standard error.
        """
        df = self.model._df_t
        idx = self.model._coefnames.index(treatment)
        se = self.model._se[idx]

        return np.sqrt((1 - r2yz_dx) / (1 - r2dz_x)) * se * np.sqrt(df / (df - 1))

    def adjusted_t(self, r2dz_x, r2yz_dx, treatment, reduce=True, h0=0):
        """
        Compute the bias-adjusted t-statistic.

        Parameters
        ----------
        r2dz_x, r2yz_dx : float or np.ndarray
            Partial R2 parameters of the confounder.
        treatment : str
            The treatment variable.
        reduce : bool, default True
            Whether to reduce the estimate magnitude.
        h0 : float, default 0
            The null hypothesis value for the t-test.

        Returns
        -------
        float or np.ndarray
            The adjusted t-statistic.
        """
        new_estimate = self.adjusted_estimate(r2dz_x, r2yz_dx, treatment = treatment, reduce = reduce)
        new_se = self.adjusted_se(r2dz_x, r2yz_dx, treatment = treatment)
        return (new_estimate - h0) / new_se

    def summary(self, treatment=None, benchmark_covariates=None, kd=[1, 2, 3], ky=None, q=1, alpha=0.05, reduce=True, decimals=3):
        """
        Print a summary of the sensitivity analysis.

        Parameters
        ----------
        treatment : str
            The name of the treatment variable. If None, defaults to the first variable.
        benchmark_covariates : list or str, optional
            The list of covariates to use for bounding. If provided, the bounds table is printed.
        kd : list, optional
            Multipliers for the strength of the confounder with the treatment (default [1, 2, 3]).
        ky : list, optional
            Multipliers for the strength of the confounder with the outcome (default same as kd).
        q : float
            The percent reduction in the estimate considered problematic (default 1 = 100%).
        alpha : float
            Significance level for the Robustness Value (default 0.05).
        reduce : bool
            Whether the bias reduces the absolute value of the estimate (default True).
        decimals : int
            Number of decimal places to print.
        """
        if treatment is None:
            treatment = self.model._coefnames[0]

        if ky is None:
            ky = kd

        formula = self.model._fml
        stats = self.sensitivity_stats(X=treatment, q=q, alpha=alpha)
        est = stats['estimate']
        se = stats['se']
        t_stat = est / se
        partial_r2 = stats['partial_R2']
        rv_q = stats['rv_q']
        rv_qa = stats['rv_qa']

        print("Sensitivity Analysis to Unobserved Confounding\n")
        print(f"Model Formula: {formula}\n")

        print(f"Null hypothesis: q = {q} and reduce = {reduce} ")

        h0 = (1 - q) * est
        print(f"-- The null hypothesis deemed problematic is H0:tau = {h0:.{decimals}f} \n")

        print(f"Unadjusted Estimates of '{treatment}':")
        print(f"  Coef. estimate: {est:.{decimals}f}")
        print(f"  Standard Error: {se:.{decimals}f}")
        print(f"  t-value: {t_stat:.{decimals}f} \n")

        print("Sensitivity Statistics:")
        print(f"  Partial R2 of treatment with outcome: {partial_r2:.{decimals}f}")
        print(f"  Robustness Value, q = {q} : {rv_q:.{decimals}f}")
        print(f"  Robustness Value, q = {q} alpha = {alpha} : {rv_qa:.{decimals}f} \n")

        if benchmark_covariates is not None:
            print("Bounds on omitted variable bias:")

            bounds_df = self.ovb_bounds(
                treatment=treatment,
                benchmark_covariates=benchmark_covariates,
                kd=kd,
                ky=ky,
                alpha=alpha,
                adjusted_estimate=True
            )

            cols = ['bound_label', 'r2dz_x', 'r2yz_dx', 'treatment',
                    'adjusted_estimate', 'adjusted_se', 'adjusted_t',
                    'adjusted_lower_CI', 'adjusted_upper_CI']

            cols = [c for c in cols if c in bounds_df.columns]

            print(bounds_df[cols].to_string(index=True, float_format=lambda x: f"{x:.{6}f}" if abs(
                x) < 1e-3 else f"{x:.{decimals}f}"))