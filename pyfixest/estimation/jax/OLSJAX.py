from typing import Optional

import jax
import jax.numpy as jnp
import pandas as pd

from pyfixest.estimation.jax.demean_jax_ import demean_jax

class OLSJAX:
    def __init__(
        self,
        X: jax.Array,
        Y: jax.Array,
        fe: Optional[jax.Array] = None,
        weights: Optional[jax.Array] = None,
        vcov: Optional[str, dict[str,str]] = None,
    ):

        """
        Class to run OLS regression in JAX.

        Parameters
        ----------
        X : jax.Array
            N x k matrix of independent variables.
        Y : jax.Array
            Dependent variable. N x 1 matrix.
        fe : jax.Array, optional
            Fixed effects. N x 1 matrix of integers. The default is None.
        weights: jax.Array, optional
            Weights. N x 1 matrix. The default is None.
        vcov : str, optional
            Type of covariance matrix. The default is None. Options are:
            - "iid" (default): iid errors
            - "HC1": heteroskedasticity robust
            - "HC2": heteroskedasticity robust
            - "HC3": heteroskedasticity robust
            - "CRV1": cluster robust. In this case, please provide a dictionary
                    with the cluster variable as key and the name of the cluster variable as value.
        """

        self.X_orignal = X
        self.Y_orignal = Y
        self.fe = fe
        self.N = X.shape[0]
        self.k = X.shape[1]
        self.weights = jnp.ones(self.N) if weights is None else weights
        self.vcov_type = "iid" if vcov is None else vcov

    def fit(self):
        self.Y, self.X = self.demean(
            Y=self.Y_orignal, X=self.X_orignal, fe=self.fe, weights=self.weights
        )
        self.beta = jnp.linalg.lstsq(self.X, self.Y)[0]
        self.residuals
        self.scores
        self.vcov(vcov_type=self.vcov_type)
        self.inference()

    @property
    def residuals(self):
        self.uhat = self.Y - self.X @ self.beta
        return self.uhat

    def vcov(self, vcov_type: str):
        bread = self.bread
        meat = self.meat(type=vcov_type)
        if vcov_type == "iid":
            self.vcov = bread * meat
        else:
            self.vcov = bread @ meat @ bread

        return self.vcov

    @property
    def bread(self):
        return jnp.linalg.inv(self.X.T @ self.X)

    @property
    def leverage(self):
        return jnp.sum(self.X * (self.X @ jnp.linalg.inv(self.X.T @ self.X)), axis=1)

    @property
    def scores(self):
        return self.X * self.residuals

    def meat(self, type: str):
        if type == "iid":
            return self.meat_iid
        elif type == "HC1":
            return self.meat_hc1
        elif type == "HC2":
            return self.meat_hc2
        elif type == "HC3":
            return self.meat_hc3
        elif type == "CRV1":
            return self.meat_crv1
        else:
            raise ValueError("Invalid type")

    @property
    def meat_iid(self):
        return jnp.sum(self.uhat**2) / (self.N - self.k)

    @property
    def meat_hc1(self):
        return self.scores.T @ self.scores

    def meat_hc2(self):
        self.leverage
        transformed_scores = self.scores / jnp.sqrt(1 - self.leverage)
        return transformed_scores.T @ transformed_scores

    def meat_hc3(self):
        self.leverage
        transformed_scores = self.scores / (1 - self.leverage)
        return transformed_scores.T @ transformed_scores

    @property
    def meat_crv1(self):
        raise NotImplementedError("CRV1 is not implemented")

    def predict(self, X):
        X = jnp.array(X)
        return X @ self.beta

    def demean(self, Y: jax.Array, X: jax.Array, fe: jax.Array, weights: jax.Array):
        if fe is not None:
            if not jnp.issubdtype(fe.dtype, jnp.integer):
                raise ValueError("Fixed effects must be integers")

            YX = jnp.concatenate((Y, X), axis=1)
            YXd, success = demean_jax(
                x=YX, flist=fe, weights=self.weights, output="jax"
            )
            Yd = YXd[:, 0].reshape(-1, 1)
            Xd = YXd[:, 1:]

            return Yd, Xd

        else:
            return Y, X

    def inference(self):
        self.se = jnp.sqrt(jnp.diag(self.vcov)).reshape(-1, 1)
        self.tstat = self.beta / self.se
        self.pvalue = 2 * (1 - jax.scipy.stats.norm.cdf(jnp.abs(self.tstat)))
        self.confint = jnp.column_stack(
            [
                self.beta - jax.scipy.stats.norm.ppf(1 - 0.05 / 2) * self.se,
                self.beta + jax.scipy.stats.norm.ppf(1 - 0.05 / 2) * self.se,
            ]
        )

        return self.se, self.tstat, self.pvalue, self.confint
