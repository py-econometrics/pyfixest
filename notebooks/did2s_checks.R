setwd("~/Documents/pyfixest/")
library(data.table)
library(fixest)
library(haven)

source("./notebooks/did2_program.R")
source("./notebooks/did2s_utils.R")

df_castle <- read_dta("https://github.com/scunning1975/mixtape/raw/master/castle.dta")

fit_unwgt = did2s(data=df_castle, 
      yname="l_homicide",
      first_stage=~ 0 | sid + year,
      second_stage=~ i(post, ref=0),
      treatment="post",
      cluster_var="state"
) 

## debug(did2s)
## debug(did2s_estimate)
      
fit_wgt = did2s(data=df_castle, 
      yname="l_homicide",
      first_stage=~ 0 | sid + year,
      second_stage=~ i(post, ref=0),
      treatment="post",
      cluster_var="state",
      weights="popwt"
) 

## Simple feols
fit_twfe = feols(l_homicide ~ post | state + year, 
      data=df_castle, 
      weights=df_castle[["popwt"]])

summary(fit_twfe)

  fit_unwgt = feols(l_homicide ~ post | state + year, 
      data=df_castle, 
)

summary(fit_unwgt)


# Compare first-stage residuals with and without weighting 
df_untreated = df_castle[df_castle[["post"]] == 0, ]

fit_untrt = feols(l_homicide ~ 1 | state + year, data=df_untreated)

fs_u = df_castle[["l_homicide"]] - predict(fit_untrt, newdata=df_castle)

# Slight difference with weighted
fs_u[1:5]

df_untreated = df_castle[df_castle[["post"]] == 0, ]

fit_untrt = feols(l_homicide ~ 1 | state + year, 
                  data=df_untreated, 
                  weights=df_untreated[["popwt"]])

fs_u = df_castle[["l_homicide"]] - predict(fit_untrt, newdata=df_castle)

fs_u[1:5]

