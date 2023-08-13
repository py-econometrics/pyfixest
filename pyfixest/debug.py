from pyfixest.fixest import Fixest
from pyfixest.utils import get_data

import pandas as pd

# import pdb; pdb.set_trace()

N = 100
seed = 879111
beta_type = "1"
error_type = "1"
model = "Fepois"
dropna = False
vcov = {'CRV1':'group_id'}
fml = "Y~X1|f2^f3^f1"

data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type, model = model)

if dropna:
    data = data.dropna()

fixest = Fixest(data=data)
fixest.fepois(fml, vcov=vcov)
