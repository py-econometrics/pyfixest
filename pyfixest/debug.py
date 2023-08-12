from pyfixest.fixest import Fixest
from pyfixest.utils import get_data

import pandas as pd
# import pdb; pdb.set_trace()

N = 1000
seed = 1234
beta_type = "1"
error_type = "1"
dropna = False

data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type)

if dropna:
    data = data.dropna()

#data["f2"] = pd.Categorical(data["f2"].astype(str))

fml = "Y + Y2 ~ sw(X1, X2) |f1"

fixest = Fixest(data=data)
fixest.feols(fml, vcov="iid")
