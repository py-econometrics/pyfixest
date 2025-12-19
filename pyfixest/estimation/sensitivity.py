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


   # let's start with R_2
   def partial_r2(self, X: Optional[str] = None) -> np.ndarray:
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
           r2 = tstat[idx]**2 / (tstat[idx]**2 + df[X])
           return r2

