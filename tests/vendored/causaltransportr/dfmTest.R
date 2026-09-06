# Vendored reference implementation for the DFM heterogeneity test.
#
# Source : https://github.com/Netflix-Skunkworks/causaltransportr
#          R/dfmTest.R (Netflix-Skunkworks/causaltransportr, main branch)
# License: MIT (Copyright 2022 Netflix, Inc.) -- see LICENSE in this directory.
#
# Committed verbatim so tests/test_dfm_test_vs_r.py can validate the Python port
# in pyfixest/estimation/post_estimation/dfm_test.py against the original. Pure
# base R (lm.fit, cov, solve, crossprod, pchisq): no package dependencies.

#' Omnibus test of systematic treatment effect heterogeneity (Ding, Feller, Miratrix 2018)
#' @description
#' Omnibus test of treatment effect heterogeneity along specified dimensions X.
#' Implemented by regressing individual treatment effect on covariates and
#' joint test of covariate coefficients = 0
#' @param X covariate matrix
#' @param a treament vector
#' @param y outcome vector
#' @return vector aith chi-squared statistic and p-value for systematic heterogeneity
#' @export
#' @references Ding, P., A. Feller, and L. Miratrix. (2019): "Decomposing Treatment
#' Effect Variation," Journal of the American Statistical Association, 114, 304-17.

dfmTest = function(y, a, X) {
  if (sum(X[, 1]) != nrow(X)) X = cbind(1, X)
  n1 = sum(a); n0 = sum(1 - a); K = ncol(X)
  # separate outcome models for each treatment level
  m1 = lm.fit(X[a == 1, ], y[a == 1]); m0 = lm.fit(X[a == 0, ], y[a == 0])
  E1 = m1$residuals * X[a == 1, ];   E0 = m0$residuals * X[a == 0, ]
  # vcov
  Sxx1Inv = solve(crossprod(X[a == 1, ]) / n1); Sxx0Inv = solve(crossprod(X[a == 0, ]) / n0)
  # projection of effect heterogeneity on covariates
  betaHat = m1$coefficients - m0$coefficients
  covBeta = (Sxx1Inv %*% (cov(E1) / n1) %*% Sxx1Inv + Sxx0Inv %*% (cov(E0) / n0) %*% Sxx0Inv)
  beta1Hat = betaHat[2:K];
  covBeta1 = covBeta[2:K, 2:K]
  # joint test on non-intercept subvector
  chisq.stat = t(beta1Hat) %*% solve(covBeta1, beta1Hat)
  chisq.pv = pchisq(chisq.stat, df = K - 1, lower.tail = FALSE)
  c(chisq.stat, chisq.pv)
}
