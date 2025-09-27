from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# rpy2 imports
from rpy2.robjects.packages import importr
