import itertools
import warnings
from dataclasses import dataclass, field
from statistics import kde_random
from typing import Any, Optional, Union
from pyfixest.estimation.estimation import feols

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.sparse import diags, hstack, spmatrix, vstack
from scipy.sparse.linalg import lsqr
from scipy.stats import t
from tqdm import tqdm

@dataclass
class SensitivityAnalysis:
    """
    Implements the sensitivity analysis method described in Cinelli and Hazlett (2020): "Making Sense of Sensitivity: Extending Omitted Variable Bias".

    This class performs the analysis, creates the benchmarks and supports visualizations and output creation.

    Parameters
    ----------
    To be added.
    """
   # Core Inputs - LIST IN PROGRESS
    model: Any
    X: Optional[str] = None

   # let's start with R_2
    def partial_r2(self, X: Optional[str] = None) -> Union[float, np.ndarray]:
       """
       Calculate the partial R2 for a given variable.

       The partial R2 explains how much of the residual variance of the outcome is explained by the covariate.
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
        Compute the robustness value of the regression coefficient.
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

        Returns the RV, partial R2 and partial f2.
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
    def ovb_bounds(self, treatment, benchmark_covariates, kd, ky):
        """
        Compute bounds on omitted variable bias using observed covariates as benchmarks.

        Parameters
        ----------
        Self
        Benchmark Covariates
        kd
        ky
        adjusted_estimate: bool
        bound_type: str
        """
        if ky is None:
            ky = kd

        if bound != "partial r2":
            sys.exit('Only partial r2 is implemented as of now.')

        bounds = self._ovb_bounds_partial_r2(model = model, treatment = treatment, benchmark_covariates = benchmark_covariates, kd = kd, ky = ky)

        if adjusted_estimate:
            bounds['treatment'] = treatment
            bounds['adjusted_estimate'] = self.adjusted_estimate(bounds['r2dz_x'], bounds['r2yz_dx'], reduce = True)
            bounds['adjusted_se'] = self.adjusted_se(bounds['r2dz_x'], bounds['r2yz_dx'])
            bounds['adjusted_t'] = self.adjusted_t(bounds['r2dz_x'], bounds['r2yz_dx'], reduce = True, h0 = 0)
            bounds['adjusted_lower_CI'] = bounds['adjusted_estimate'] - se_multiple * bounds['adjusted_se']
            bounds['adjusted_upper_CI'] = bounds['adjusted_estimate'] + se_multiple * bounds['adjusted_se']

        return bounds

    def _ovb_bounds_partial_r2(self, treatment, benchmark_covariates, kd, ky):
        """
        Compute OVB bounds based on partial R2.

        This function should not be called directly. It is called under the ovb_bounds user facing function.
        """
        model = self.model
        if (model is None or treatment is None):
            sys.exit('ovb_partial_r2 requires a model object and a treatment variable')

        data = model._data
        X = model._X
        fixef = model._fixef

        non_treatment = X.drop(columns = treatment)
        covariate_names = non_treatment.columns.tolist()
        covariates = ' + '.join(covariate_names)

        if fixef == "0":
            formula = f"{treatment} ~ {covariates}"
        else:
            formula = f"{treatment} ~ {covariates} | {fixef}"

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
                    raise ValueError(f"Implied bound on r2dz.x >= 1 for benchmark {b} with kd={k_d_val}."
                                     "Impossible scenario. Try a lower kd.")
                r2zxj_xd = kd_val * (r2dxj_x**2) / ((1 - kd_val * r2dxj_x) * (1 - r2dxj_x))

                if r2zxj_xd >= 1:
                    raise ValueError(f"Impossible kd value for benchmark {b}. Try a lower kd.")

                r2yz_dx = (((np.sqrt(ky_val) + np.sqrt(r2zxj_xd)) / np.sqrt(1 - r2zxj_xd))**2) * (r2yxj_dx / (1 - r2yxj_dx))

                if r2yz_dx > 1:
                    print(f"Warning: Implied bound on r2yz.dx > 1 for {b}. Capping at 1.")
                    r2yz_dx = 1.0

                bounds_list.append({
                    'bound_label': f"{k_d_val}x {b}",  # Simple label maker
                    'r2dz_x': r2dz_x,
                    'r2yz_dx': r2yz_dx,
                    'benchmark_covariate': b,
                    'kd': k_d_val,
                    'ky': k_y_val
                })

        return pd.DataFrame(bounds_list)

    def bias(self, r2dz_x, r2yz_dx):
        """
        Compute the bias for the partial R2 parametrization.
        """
        df = self.model._df_t
        se = self.model._se

        r2dz_x, r2yz_dx = np.array(r2dz_x), np.array(r2yz_dx)
        bias_factor = np.sqrt((r2yz_dx * r2dz_x) / (1 - r2dz_x))

        return bias_factor * se * np.sqrt(df)

    def adjusted_estimate(self, r2dz_x, r2yz_dx, reduce=True):
        """
        Compute the bias-adjusted coefficient estimate.
        """
        estimate = self.model._beta_hat
        if reduce:
            return np.sign(estimate) * (abs(estimate) - self.bias(r2dz_x, r2yz_dx))
        else:
            return np.sign(estimate) * (abs(estimate) + self.bias(r2dz_x, r2yz_dx))

    def adjusted_se(self, r2dz_x, r2yz_dx):
        """
        Compute the bias-adjusted Standard Error estimate.
        """
        df = self.model._df_t
        se = self.model._se

        return np.sqrt((1 - r2yz_dx) / (1 - r2dz_x)) * se * np.sqrt(df / (df - 1))

    def adjusted_t(self, r2dz_x, r2yz_dx, reduce=True, h0=0):
        """
        Compute the bias-adjusted t-statistic.
        """
        new_estimate = self.adjusted_estimate(r2dx_x, r2yz_dx, reduce = reduce)
        new_se = self.adjusted_se(r2dx_x, r2yz_dx)
        return (new_estimate - h0) / new_se
