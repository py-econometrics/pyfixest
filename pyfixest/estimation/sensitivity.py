import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Union

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
    Implements the sensitivity analysis method described in Cinelli and Hazlett (2020): "Making Sense of Sensitivity: Extending Omitted Variable Bias"

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
       Calculates the partial R2 for a given variable. The partial R2 explains how much of the residual variance of the outcome is explained by the covariate.
       """
       df = self.model._df_t
       names = self.model._coefnames
       tstat = self.model._tstat

       if X is None:
           r2 = tstat**2 / (tstat**2 + df)
           return r2
       else:
           idx = names.index(X)
           r2 = tstat[idx]**2 / (tstat[idx]**2 + df)
           return r2

    # define partial f2
    def partial_f2(self, X: Optional[str] = None) -> Union[float, np.ndarray]:
        """
        Compute the partial (Cohen's) f2 for a linear regression model. The partial f2 is a measure of effect size (a transformation of the partial R2).
        """
        df = self.model._df_t
        names = self.model._coefnames
        tstat = self.model._tstat

        if X is None:
            f2 = tstat**2 / df
            return f2
        else:
            idx = names.index(X)
            f2 = tstat[idx]**2 / df
            return f2

    # robustness value function
    def robustness_value(self, X: Optional[str] = None, q = 1, alpha = 1.0) -> Union[float, np.ndarray]:
        """
        Compute the robustness value of the regression coefficient
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

    # summary function to report these
    def sensitivity_stats(self, X: Optional[str] = None, q = 1, alpha = 0.05) -> dict:
        """
        Compute the sensitivity statistics for the model. Returns the RV, partial R2 and partial f2
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
