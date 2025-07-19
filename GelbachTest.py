%pip install git+https://github.com/Erica-Ryan/pyfixest.git

import pyfixest as pft
import pandas as pd
import numpy as np
from scipy import stats
from patsy import Treatment
import re
import numexpr
pd.options.mode.chained_assignment = None
np.random.seed(42)

# Create 2000 observations
n = 2000

# Create the dataframe
df = pd.DataFrame({
    'var_y': np.random.uniform(0, 1, n),  # Random values between 0 and 1
    'var_decomp': np.random.choice(['cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_4'], n),  # Random selection from categories
    'var_x1': np.random.choice([0, 1, 2, 5, 8, 9], n),  # Random continuous control variable
    'var_x2_1': np.random.uniform(0, 1, n),  # Random continuous x2 variable
    'var_x2_2': np.random.uniform(0, 1, n),  # Random continuous x2 variable
    'var_x2_3': np.random.uniform(0, 1, n),  # Random continuous x2 variable
    'var_x2_4': np.random.choice([0, 1], n)  # Random categorical x2 variable
})

# Create indicator variable for reference category
df['cat_0_indicator'] = (df['var_decomp'] == 'cat_0').astype(int)

# Create indicator variables for decomp variable
decomp_dummies = pd.get_dummies(df['var_decomp'], prefix='var_decomp').astype(int)

# Add these new columns to the original dataframe
df = pd.concat([df, decomp_dummies], axis=1)

fit_full = pft.feols(
    "var_y ~ var_x2_4 + var_x1 + var_x2_1",
    data=df,
)
fit_full.summary()

fit_full.decompose(param="var_x2_4", digits=5)

fit_full.decompose_x1(decomp_var="var_x2_4", x1_vars = ['var_x2_4'], digits=5)
