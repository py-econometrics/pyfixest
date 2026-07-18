# Replicating Examples from “The Effect”

This notebook replicates code examples from Nick Huntington-Klein’s book on causal inference, [The Effect](https://theeffectbook.net/).

``` python
from causaldata import Mroz, gapminder, organ_donations, restaurant_inspections

import pyfixest as pf

%load_ext watermark
%watermark --iversions
```

    causaldata: 0.1.5
    pyfixest  : 0.40.1

## Chapter 4: Describing Relationships

``` python
# Read in data
dt = Mroz.load_pandas().data
# Keep just working women
dt = dt.query("lfp")
# Create unlogged earnings
dt.loc[:, "earn"] = dt["lwg"].apply("exp")

# 5. Run multiple linear regression models by succesively adding controls
fit = pf.feols(fml="lwg ~ csw(inc, wc, k5)", data=dt, vcov="iid")
pf.etable(fit)
```

    /var/folders/98/c353q4p95v5_fz62gr5c9td80000gn/T/ipykernel_48960/786816010.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      dt.loc[:, "earn"] = dt["lwg"].apply("exp")

[TABLE]

## Chapter 13: Regression

### Example 1

``` python
res = restaurant_inspections.load_pandas().data
res.inspection_score = res.inspection_score.astype(float)
res.NumberofLocations = res.NumberofLocations.astype(float)
res.dtypes

fit = pf.feols(fml="inspection_score ~ NumberofLocations", data=res)
pf.etable([fit])
```

[TABLE]

### Example 2

``` python
df = restaurant_inspections.load_pandas().data

fit1 = pf.feols(
    fml="inspection_score ~ NumberofLocations + I(NumberofLocations^2) + Year", data=df
)
fit2 = pf.feols(fml="inspection_score ~ NumberofLocations*Weekend + Year", data=df)

pf.etable([fit1, fit2])
```

[TABLE]

### Example 3: HC Standard Errors

``` python
pf.feols(fml="inspection_score ~ Year + Weekend", data=df, vcov="HC3").summary()
```

    ###

    Estimation:  OLS
    Dep. var.: inspection_score
    Inference:  HC3
    Observations:  27178

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |    2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|--------:|--------:|
    | Intercept     |    185.380 |       12.150 |    15.257 |      0.000 | 161.564 | 209.196 |
    | Year          |     -0.046 |        0.006 |    -7.551 |      0.000 |  -0.057 |  -0.034 |
    | Weekend       |      2.057 |        0.353 |     5.829 |      0.000 |   1.365 |   2.749 |
    ---
    RMSE: 6.248 R2: 0.003 

### Example 4: Clustered Standard Errors

``` python
pf.feols(
    fml="inspection_score ~ Year + Weekend", data=df, vcov={"CRV1": "Weekend"}
).tidy()
```

|  | Estimate | Std. Error | t value | Pr(\>\|t\|) | 2.5% | 97.5% |
|----|----|----|----|----|----|----|
| Coefficient |  |  |  |  |  |  |
| Intercept | 185.380033 | 3.264345 | 56.789344 | 0.011209 | 143.902592 | 226.857474 |
| Year | -0.045640 | 0.001624 | -28.107556 | 0.022640 | -0.066272 | -0.025008 |
| Weekend | 2.057166 | 0.001401 | 1468.256802 | 0.000434 | 2.039364 | 2.074969 |

### Example 5: Bootstrap Inference

``` python
fit = pf.feols(fml="inspection_score ~ Year + Weekend", data=df)
fit.wildboottest(reps=999, param="Year")
```

    param                 Year
    t value           -7.55233
    Pr(>|t|)               0.0
    bootstrap_type          11
    inference               HC
    impose_null           True
    ssc               1.000074
    dtype: object

## Chapter 16: Fixed Effects

### Example 1

tba

### Example 2

``` python
gm = gapminder.load_pandas().data
gm["logGDPpercap"] = gm["gdpPercap"].apply("log")

fit = pf.feols(fml="lifeExp ~ C(country) + np.log(gdpPercap)", data=gm)
fit.tidy().head()
```

|  | Estimate | Std. Error | t value | Pr(\>\|t\|) | 2.5% | 97.5% |
|----|----|----|----|----|----|----|
| Coefficient |  |  |  |  |  |  |
| Intercept | -27.773459 | 2.500533 | -11.107015 | 0.000000e+00 | -32.678217 | -22.868701 |
| C(country)\[T.Albania\] | 17.782625 | 2.195160 | 8.100835 | 1.110223e-15 | 13.476853 | 22.088397 |
| C(country)\[T.Algeria\] | 5.241055 | 2.214496 | 2.366704 | 1.806875e-02 | 0.897356 | 9.584755 |
| C(country)\[T.Angola\] | -13.907122 | 2.201727 | -6.316460 | 3.481857e-10 | -18.225777 | -9.588468 |
| C(country)\[T.Argentina\] | 8.132158 | 2.272781 | 3.578065 | 3.567229e-04 | 3.674133 | 12.590183 |

### Example 3: TWFE

``` python
# Set our individual and time (index) for our data
fit = pf.feols(fml="lifeExp ~ np.log(gdpPercap) | country + year", data=gm)
fit.summary()
```

    OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.

    ###

    Estimation:  OLS
    Dep. var.: lifeExp, Fixed effects: country + year
    Inference:  iid
    Observations:  1704

    | Coefficient       |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:------------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | np.log(gdpPercap) |      1.450 |        0.268 |     5.419 |      0.000 |  0.925 |   1.975 |
    ---
    RMSE: 3.267 R2: 0.936 R2 Within: 0.019 

## Chapter 18: Difference-in-Differences

### Example 1

``` python
od = organ_donations.load_pandas().data

# Create Treatment Variable
od["California"] = od["State"] == "California"
od["After"] = od["Quarter_Num"] > 3
od["Treated"] = 1 * (od["California"] & od["After"])

did = pf.feols(fml="Rate ~ Treated | State + Quarter", data=od)
did.summary()
```

    ###

    Estimation:  OLS
    Dep. var.: Rate, Fixed effects: State + Quarter
    Inference:  iid
    Observations:  162

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | Treated       |     -0.022 |        0.020 |    -1.096 |      0.275 | -0.063 |   0.018 |
    ---
    RMSE: 0.022 R2: 0.979 R2 Within: 0.009 

### Example 3: Dynamic Treatment Effect

``` python
od = organ_donations.load_pandas().data

# Create Treatment Variable
od["California"] = od["State"] == "California"
# od["Quarter_Num"] = pd.Categorical(od.Quarter_Num)
od["California"] = od.California.astype(float)

did2 = pf.feols(
    fml="Rate ~ i(Quarter_Num, California,ref=3) | State + Quarter_Num", data=od
)

did2.tidy()
```

|  | Estimate | Std. Error | t value | Pr(\>\|t\|) | 2.5% | 97.5% |
|----|----|----|----|----|----|----|
| Coefficient |  |  |  |  |  |  |
| Quarter_Num::1:California | -0.002942 | 0.036055 | -0.081606 | 0.935090 | -0.074299 | 0.068415 |
| Quarter_Num::2:California | 0.006296 | 0.036055 | 0.174627 | 0.861655 | -0.065061 | 0.077653 |
| Quarter_Num::4:California | -0.021565 | 0.036055 | -0.598127 | 0.550837 | -0.092922 | 0.049792 |
| Quarter_Num::5:California | -0.020292 | 0.036055 | -0.562817 | 0.574567 | -0.091649 | 0.051065 |
| Quarter_Num::6:California | -0.022165 | 0.036055 | -0.614768 | 0.539825 | -0.093522 | 0.049192 |
