from pyfixest.fixest import Fixest
from pyfixest.utils import get_data

# import pdb; pdb.set_trace()

N = 100
seed = 1234
beta_type = "1"
error_type = "1"

data = get_data(N=N, seed=seed, beta_type=beta_type, error_type=error_type)
data.head()

fml = "log(Y) ~ log(X1):X2 | f3 + f1"

fixest = Fixest(data=data)
fixest.feols(fml, vcov="iid")
