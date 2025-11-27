# Examples of using custom fixed effects markers in pyfixest etable

import pyfixest as pf

# Load data
df = pf.get_data()
fit1 = pf.feols("Y~X1 + X2 | f1", df)
fit2 = pf.feols("Y~X1 + X2 | f1 + f2", df)

# Default behavior (x and -)
pf.etable([fit1, fit2])

# Use YES/NO
pf.etable([fit1, fit2], fe_present="YES", fe_absent="NO")

# Use Y/N
pf.etable([fit1, fit2], fe_present="Y", fe_absent="N")

# Use checkmark and cross
pf.etable([fit1, fit2], fe_present="✓", fe_absent="✗")

# Use green check and cross emojis
pf.etable([fit1, fit2], fe_present="✅", fe_absent="❌")

# Use checkmark with blank for absent
pf.etable([fit1, fit2], fe_present="✓", fe_absent="")

# Custom text markers
pf.etable([fit1, fit2], fe_present="Included", fe_absent="Excluded")
