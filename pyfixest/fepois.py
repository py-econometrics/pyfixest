import numpy as np
import pandas as pd
import warnings

from importlib import import_module
from typing import Union, List, Dict
from scipy.stats import norm, t
from pyfixest.ssc_utils import get_ssc


class Fepois:

    '''
    A class that implements the fixed effects Poisson estimator.

    Args
    '''

    def __init__(self, Y: np.ndarray, X: np.ndarray, fe: np.ndarray) -> None:

        self.Y = Y
        self.X = X

        self.N, self.k = X.shape


    def get_fit(self):

        '''
        Get the Poisson Regression Estimate for a single model. Implements the algorithm from
        Stata's pplmhdfe routine.
        '''

        pass

