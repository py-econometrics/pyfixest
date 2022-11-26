import pytest
import numpy as np
import pandas as pd
from pyfixest import feols

# rpy2 imports
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

fwildclusterboot = importr("fwildclusterboot")
stats = importr('stats')

