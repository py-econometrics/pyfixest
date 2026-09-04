from __future__ import annotations

import re
import warnings
from importlib import import_module
from typing import Any, ClassVar, Literal, cast, overload

import formulaic
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.stats import t

from pyfixest.estimation.capabilities import (
    DIFFERENCE_IN_DIFFERENCES,
    FIXED_EFFECTS,
    FREQUENCY_WEIGHTS,
    NON_FREQUENCY_WEIGHTS,
    WEIGHTED,
    Capabilities,
    supported,
    unless,
)
from pyfixest.estimation.formula import FORMULAIC_TRANSFORMS
from pyfixest.estimation.internals.fit_ import fit_ols
from pyfixest.estimation.internals.literals import (
    EstimatorKind,
    PredictionErrorOptions,
    PredictionType,
)
from pyfixest.estimation.internals.vcov_ import vcov_crv3_fast
from pyfixest.estimation.models.base_regression_ import (
    BaseRegression,
    _check_vcov_input,
    _deparse_vcov_input,
)
from pyfixest.estimation.post_estimation.decomposition import (
    GelbachDecomposition,
    _decompose_arg_check,
)

# Re-exported for backward compatibility: `pyfixest.estimation.feols_` has
# always published the covariance-input helpers and the prediction literals
# from this module.
__all__ = [
    "Feols",
    "PredictionErrorOptions",
    "PredictionType",
    "_check_vcov_input",
    "_deparse_vcov_input",
    "decomposition_type",
    "prediction_type",
]

decomposition_type = Literal["gelbach"]
prediction_type = Literal["response", "link"]


class Feols(BaseRegression):
    """
    Non user-facing class to estimate a linear regression via OLS.

    Users should not directly instantiate this class,
    but rather use the [feols()](/reference/estimation.api.feols.feols.qmd)
    function. This class constructs within-scale arrays through a shared
    ``DemeanCache``; the estimation runner supplies that cache for reuse across
    multiple fits.

    `Feols` is the ordinary-least-squares leaf of
    [BaseRegression](/reference/estimation.models.base_regression_.BaseRegression.qmd).
    It adds the OLS solve, the goodness-of-fit and Wald statistics reported for
    a linear fit, the closed-form CRV3 jackknife, and the post-estimation
    methods whose derivation assumes a single linear estimating equation:
    `wildboottest()`, `ccv()`, `decompose()`, `update()`, `evalue()`, and
    `pvalue_savi()`. Everything else is shared with the GLM and quantile
    estimators and documented on the base class.

    See the base class for the constructor arguments and the retained
    attributes. Difference-in-differences entry points return `Feols` results
    and relabel `_method`; `test_treatment_heterogeneity()`, `aggregate()`, and
    `iplot_aggregate()` are bound by those entry points and raise otherwise.

    Examples
    --------
    ```{python}
    import pyfixest as pf

    fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
    fit.tidy()
    ```
    """

    _estimator: ClassVar[EstimatorKind] = "feols"
    # Which post-estimation features an OLS fit supports; see
    # `pyfixest.estimation.capabilities`. Argument-level restrictions stay with
    # their methods, so only fit-level support is declared here.
    _capabilities: ClassVar[Capabilities] = Capabilities(
        crv3=supported(),
        hac=unless(FREQUENCY_WEIGHTS),
        wildboottest=unless(WEIGHTED),
        ccv=unless(FIXED_EFFECTS, WEIGHTED),
        decompose=unless(NON_FREQUENCY_WEIGHTS),
        ritest=unless(DIFFERENCE_IN_DIFFERENCES, WEIGHTED),
        fixef=supported(),
        predict=supported(),
        prediction_errors=unless(FIXED_EFFECTS, WEIGHTED),
        update=unless(DIFFERENCE_IN_DIFFERENCES, FIXED_EFFECTS, WEIGHTED),
        savi=unless(DIFFERENCE_IN_DIFFERENCES, FIXED_EFFECTS, WEIGHTED),
    )

    def _bind_estimator_methods(self) -> None:
        """Bind the difference-in-differences placeholders.

        `event_study()` and `did2s()` return `Feols` results and rebind these
        names on the result they hand back. A vanilla `feols()` fit keeps the
        placeholder, which names the estimator the method belongs to.
        """

        def _not_implemented_did(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                "This method is only available for DiD models, not for vanilla 'feols'."
            )

        self.test_treatment_heterogeneity = _not_implemented_did
        self.aggregate = _not_implemented_did
        self.iplot_aggregate = _not_implemented_did

    def _get_predictors(self) -> None:
        self._Y_hat_link = self._response - self.resid()
        self._Y_hat_response = self._Y_hat_link

    def get_fit(self) -> None:
        """
        Fit an OLS model.

        Returns
        -------
        None
        """
        within_data = self._drop_multicollinear_within_data(self._prepare_within_data())
        self._set_within_data(within_data)

        if self._X_is_empty:
            self._u_hat = within_data.response.flatten()
        else:
            fit = fit_ols(
                X=within_data.design,
                Y=within_data.response,
                weights=self._observation_weights.values,
                solver=self._solver,
            )

            self._tZX = fit.tZX
            self._tZy = fit.tZy
            self._beta_hat = fit.beta
            self._u_hat = fit.residuals
            self._scores = fit.scores
            self._hessian = fit.hessian

            # IV attributes, set to None for OLS, Poisson
            self._tXZ = np.array([])
            self._tZZinv = np.array([])

        self._get_predictors()

    def _finalize_fit(self) -> None:
        """Report the goodness-of-fit and Wald statistics of a linear fit."""
        self.get_performance()
        self.wald_test()

    def _vcov_crv3(
        self,
        clustid: np.ndarray,
        cluster_col: np.ndarray,
        *,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """Take the closed-form cluster jackknife where the shortcut applies.

        `vcov_crv3_fast` evaluates the leave-one-cluster-out refits of a plain
        OLS fit algebraically. It does not describe a fit with absorbed fixed
        effects, and a difference-in-differences result carries two-step or
        event-study estimates whose refit is not the retained design's, so both
        fall back to the explicit refit.
        """
        if not self._has_fixef and not self._fit_features.is_did:
            return self._vcov_crv3_fast(clustid=clustid, cluster_col=cluster_col)
        return super()._vcov_crv3(clustid=clustid, cluster_col=cluster_col, data=data)

    def _vcov_crv3_fast(self, clustid, cluster_col):
        return vcov_crv3_fast(
            X=self._X,
            Y=self._Y,
            weights=self._observation_weights.values,
            beta_hat=self._beta_hat,
            clustid=clustid,
            cluster_col=cluster_col,
        )

    def _crv3_refit(self, data: pd.DataFrame) -> Feols:
        """Replay OLS for one leave-one-cluster-out sample."""
        # lazy loading to avoid circular import
        fixest_module = import_module("pyfixest.estimation")
        # A fitted result carries the expanded single formula and the refit
        # passes no split, so this never returns a multiple-estimation object.
        return cast(
            "Feols",
            fixest_module.feols(
                fml=self._fml,
                data=data,
                vcov="iid",
                **self._estimation_refit_kwargs(),
            ),
        )

    def wildboottest(
        self,
        reps: int,
        cluster: str | None = None,
        param: str | None = None,
        weights_type: str = "rademacher",
        impose_null: bool = True,
        bootstrap_type: str = "11",
        seed: int | None = None,
        k_adj: bool = True,
        G_adj: bool = True,
        parallel: bool = False,
        return_bootstrapped_t_stats=False,
    ):
        """
        Run a wild cluster bootstrap based on an object of type "Feols".

        Parameters
        ----------
        reps : int
            The number of bootstrap iterations to run.
        cluster : Union[str, None], optional
            The variable used for clustering. Defaults to None. If None, then
            uses the variable specified in the model's `clustervar` attribute.
            If no `_clustervar` attribute is found, runs a heteroskedasticity-
            robust bootstrap.
        param : Union[str, None], optional
            A string of length one, containing the test parameter of interest.
            Defaults to None.
        weights_type : str, optional
            The type of bootstrap weights. Options are 'rademacher', 'mammen',
            'webb', or 'normal'. Defaults to 'rademacher'.
        impose_null : bool, optional
            Indicates whether to impose the null hypothesis on the bootstrap DGP.
            Defaults to True.
        bootstrap_type : str, optional
            A string of length one to choose the bootstrap type.
            Options are '11', '31', '13', or '33'. Defaults to '11'.
        seed : Union[int, None], optional
            An option to provide a random seed. Defaults to None.
        k_adj : bool, optional
            Indicates whether to apply a small sample adjustment for the number
            of observations and covariates. Defaults to True.
        G_adj : bool, optional
            Indicates whether to apply a small sample adjustment for the number
            of clusters. Defaults to True.
        parallel : bool, optional
            Indicates whether to run the bootstrap in parallel. Defaults to False.
        seed : Union[str, None], optional
            An option to provide a random seed. Defaults to None.
        return_bootstrapped_t_stats : bool, optional:
            If True, the method returns a tuple of the regular output and the
            bootstrapped t-stats. Defaults to False.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the original, non-bootstrapped t-statistic and
            bootstrapped p-value, along with the bootstrap type, inference type
            (HC vs CRV), and whether the null hypothesis was imposed on the
            bootstrap DGP. If `return_bootstrapped_t_stats` is True, the method
            returns a tuple of the regular output and the bootstrapped t-stats.

        Examples
        --------
        ```{python}
        #| echo: true
        #| results: asis
        #| include: true

        import re
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2 | f1", data)

        fit.wildboottest(
            param = "X1",
            reps=1000,
            seed = 822
        )

        fit.wildboottest(
            param = "X1",
            reps=1000,
            seed = 822,
            bootstrap_type = "31"
        )

        ```
        """
        if param is not None and param not in self._coefnames:
            raise ValueError(
                f"Parameter {param} not found in the model's coefficients."
            )

        self._require_support("wildboottest", subject="Wild cluster bootstrap")

        self._require_fit_arrays("wildboottest", arrays="the fitted arrays")
        self._require_estimation_data("wildboottest")

        cluster_list = []

        if cluster is not None and isinstance(cluster, str):
            cluster_list = [cluster]
        if cluster is not None and isinstance(cluster, list):
            cluster_list = cluster

        if cluster is None and self._clustervar is not None:
            if isinstance(self._clustervar, str):
                cluster_list = [self._clustervar]
            else:
                cluster_list = self._clustervar

        run_heteroskedastic = not cluster_list

        if not run_heteroskedastic and not len(cluster_list) == 1:
            raise NotImplementedError(
                "Multiway clustering is currently not supported with the wild cluster bootstrap."
            )

        if not run_heteroskedastic and cluster_list[0] not in self._data.columns:
            raise ValueError(
                f"Cluster variable {cluster_list[0]} not found in the data."
            )

        try:
            from wildboottest.wildboottest import WildboottestCL, WildboottestHC
        except ImportError:
            print(
                "Module 'wildboottest' not found. Please install 'wildboottest', e.g. via `PyPi`."
            )

        _Y, _X, _xnames = self._model_matrix_one_hot()

        # later: allow r <> 0 and custom R
        R = np.zeros(len(_xnames))
        if param is not None:
            R[_xnames.index(param)] = 1
        r = 0

        if run_heteroskedastic:
            inference = "HC"

            boot = WildboottestHC(X=_X, Y=_Y, R=R, r=r, B=reps, seed=seed)
            boot.get_adjustments(bootstrap_type=bootstrap_type)
            boot.get_uhat(impose_null=impose_null)
            boot.get_tboot(weights_type=weights_type)
            boot.get_tstat()
            boot.get_pvalue(pval_type="two-tailed")
            full_enumeration_warn = False
            small_sample_correction = boot.small_sample_correction

        else:
            inference = f"CRV({cluster_list[0]})"

            cluster_array = self._data[cluster_list[0]].to_numpy().flatten()

            boot = WildboottestCL(
                X=_X,
                Y=_Y,
                cluster=cluster_array,
                R=R,
                B=reps,
                seed=seed,
                parallel=parallel,
            )
            boot.get_scores(
                bootstrap_type=bootstrap_type,
                impose_null=impose_null,
                adj=k_adj,
                cluster_adj=G_adj,
            )
            _, _, full_enumeration_warn = boot.get_weights(weights_type=weights_type)
            boot.get_numer()
            boot.get_denom()
            boot.get_tboot()
            boot.get_vcov()
            boot.get_tstat()
            boot.get_pvalue(pval_type="two-tailed")
            small_sample_correction = boot.ssc

            if full_enumeration_warn:
                warnings.warn(
                    "2^G < the number of boot iterations, setting full_enumeration to True."
                )

        if np.isscalar(boot.t_stat):
            boot.t_stat = np.asarray(boot.t_stat)
        else:
            boot.t_stat = boot.t_stat[0]

        res = {
            "param": param,
            "t value": boot.t_stat.astype(np.float64),
            "Pr(>|t|)": np.asarray(boot.pvalue).astype(np.float64),
            "bootstrap_type": bootstrap_type,
            "inference": inference,
            "impose_null": impose_null,
            "ssc": small_sample_correction,
        }

        res_df = pd.Series(res)

        if return_bootstrapped_t_stats:
            return res_df, boot.t_boot
        else:
            return res_df

    def ccv(
        self,
        treatment,
        cluster: str | None = None,
        seed: int | None = None,
        n_splits: int = 8,
        pk: float = 1,
        qk: float = 1,
    ) -> pd.DataFrame:
        """
        Compute the Causal Cluster Variance following Abadie et al (QJE 2023).

        Parameters
        ----------
        treatment: str
            The name of the treatment variable.
        cluster : str
            The name of the cluster variable. None by default.
            If None, uses the cluster variable from the model fit.
        seed : int, optional
            An integer to set the random seed. Defaults to None.
        n_splits : int, optional
            The number of splits to use in the cross-fitting procedure. Defaults to 8.
        pk: float, optional
            The proportion of sampled clusters. Defaults to 1, which
            corresponds to all clusters of the population being sampled.
        qk: float, optional
            The proportion of sampled observations within each cluster.
            Defaults to 1, which corresponds to all observations within
            each cluster being sampled.

        Returns
        -------
        pd.DataFrame
            A DataFrame with inference based on the "Causal Cluster Variance"
            and "regular" CRV1 inference.

        Examples
        --------
        ```{python}
        import pyfixest as pf
        import numpy as np

        data = pf.get_data()
        data["D"] = np.random.choice([0, 1], size=data.shape[0])

        fit = pf.feols("Y ~ D", data=data, vcov={"CRV1": "group_id"})
        fit.ccv(treatment="D", pk=0.05, qk=0.5, n_splits=8, seed=123).head()
        ```
        """
        self._require_support("ccv", subject="The causal cluster variance estimator")
        assert isinstance(treatment, str), "treatment must be a string."
        assert isinstance(cluster, str) or cluster is None, (
            "cluster must be a string or None."
        )
        assert isinstance(seed, int) or seed is None, "seed must be an integer or None."
        assert isinstance(n_splits, int), "n_splits must be an integer."
        assert isinstance(pk, (int, float)) and 0 <= pk <= 1
        assert isinstance(qk, (int, float)) and 0 <= qk <= 1

        if treatment not in self._coefnames:
            raise ValueError(
                f"Variable {treatment} not found in the model's coefficients."
            )

        self._require_fit_arrays("ccv", arrays="the fitted arrays")
        self._require_estimation_data("ccv")

        if cluster is None:
            if self._clustervar is None:
                raise ValueError("No cluster variable found in the model fit.")
            elif len(self._clustervar) > 1:
                raise ValueError(
                    "Multiway clustering is currently not supported with the causal cluster variance estimator."
                )
            else:
                cluster = self._clustervar[0]

        # check that cluster is in data
        if cluster not in self._data.columns:
            raise ValueError(
                f"Cluster variable {cluster} not found in the data used for the model fit."
            )

        if not self._is_clustered:
            warnings.warn(
                "The initial model was not clustered. CRV1 inference is computed and stored in the model object."
            )
            self.vcov({"CRV1": cluster})

        if seed is None:
            seed = np.random.randint(1, 100_000_000)
        rng = np.random.default_rng(seed)

        fml = self._fml
        data = self._data
        Y = self._Y.flatten()
        W = data[treatment].to_numpy()
        assert np.all(np.isin(W, [0, 1])), (
            "Treatment variable must be binary with values 0 and 1"
        )
        X = self._X
        cluster_vec = data[cluster].to_numpy()
        unique_clusters = np.unique(cluster_vec)

        tau_full = float(self.coef().xs(treatment))

        N = self._N
        G = len(unique_clusters)

        ccv_module = import_module("pyfixest.estimation.post_estimation.ccv")
        _compute_CCV = ccv_module._compute_CCV

        vcov_splits = 0.0
        for _ in range(n_splits):
            vcov_ccv = _compute_CCV(
                fml=fml,
                Y=Y,
                X=X,
                W=W,
                rng=rng,
                data=data,
                treatment=treatment,
                cluster_vec=cluster_vec,
                pk=pk,
                tau_full=tau_full,
                demeaner=self._demeaner,
            )
            vcov_splits += vcov_ccv

        vcov_splits /= n_splits
        vcov_splits /= N

        crv1_idx = self._coefnames.index(treatment)
        vcov_crv1 = self._vcov[crv1_idx, crv1_idx]
        vcov_ccv = qk * vcov_splits + (1 - qk) * vcov_crv1

        se = np.sqrt(vcov_ccv)
        tstat = tau_full / se
        df = G - 1
        pvalue = 2 * (1 - t.cdf(np.abs(tstat), df))
        alpha = 0.95
        z = np.abs(t.ppf((1 - alpha) / 2, df))
        z_se = z * se
        conf_int = np.array([tau_full - z_se, tau_full + z_se])

        res_ccv_dict: dict[str, float | np.ndarray] = {
            "Estimate": tau_full,
            "Std. Error": se,
            "t value": tstat,
            "Pr(>|t|)": pvalue,
            "2.5%": conf_int[0],
            "97.5%": conf_int[1],
        }

        res_ccv = pd.Series(res_ccv_dict)

        res_ccv.name = "CCV"

        res_crv1 = cast(pd.Series, self.tidy().xs(treatment))
        res_crv1.name = "CRV1"

        return pd.concat([res_ccv, res_crv1], axis=1).T

    @overload
    def _model_matrix_one_hot(
        self, output: Literal["numpy"] = "numpy"
    ) -> tuple[np.ndarray, np.ndarray, list[str]]: ...

    @overload
    def _model_matrix_one_hot(
        self, output: Literal["sparse"]
    ) -> tuple[np.ndarray, csc_matrix, list[str]]: ...

    def _model_matrix_one_hot(
        self, output: Literal["numpy", "sparse"] = "numpy"
    ) -> tuple[np.ndarray, np.ndarray | csc_matrix, list[str]]:
        """
        Transform a model matrix with fixed effects into a one-hot encoded matrix.

        Parameters
        ----------
        output : str, optional
            The type of output. Defaults to "numpy", in which case the returned matrices
            Y and X are numpy arrays. If set to "sparse", the returned design matrix X will
            be sparse.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, list[str]]
            A tuple with the dependent variable, the model matrix, and the column names.
        """
        if self._has_fixef:
            fml_linear, fixef = self._fml.split("|")
            fixef_vars = fixef.split("+")
            fixef_vars_C = [f"C({x})" for x in fixef_vars]
            fixef_fml = "+".join(fixef_vars_C)
            fml_dummies = f"{fml_linear} + {fixef_fml}"
            # output = "pandas" as Y, X need to be np.arrays for parallel processing
            # if output = "numpy", type of Y, X is not np.ndarray but a formulaic object
            # which cannot be pickled by joblib

            Y, X = formulaic.Formula(fml_dummies).get_model_matrix(
                self._data,
                output=output,
                context=FORMULAIC_TRANSFORMS | {**self._context},
            )
            xnames = X.model_spec.column_names
            Y = Y.toarray().flatten() if output == "sparse" else Y.flatten()
            X = csc_matrix(X) if output == "sparse" else X

        else:
            Y = self._Y.flatten()
            X = self._X
            xnames = self._coefnames

        X = csc_matrix(X) if output == "sparse" else X

        return Y, X, xnames

    def decompose(
        self,
        param: str | None = None,
        x1_vars: list[str] | str | None = None,
        decomp_var: str | None = None,
        type: decomposition_type = "gelbach",
        cluster: str | None = None,
        combine_covariates: dict[str, list[str]] | None = None,
        reps: int = 1000,
        seed: int | None = None,
        nthreads: int | None = None,
        agg_first: bool | None = None,
        only_coef: bool = False,
        digits=4,
    ) -> GelbachDecomposition:
        """
        Implement the Gelbach (2016) decomposition method for mediation analysis.

        Compares a short model `depvar on param` with the long model
        specified in the original feols() call.

        For details, take a look at
        "When do covariates matter?" by Gelbach (2016, JoLe). You can find
        an ungated version of the paper on SSRN under the following link:
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1425737 .

        When the initial regression is weighted, weights are interpreted as frequency
        weights. Inference is not yet supported for weighted models.

        Parameters
        ----------
        param : str
            The name of the focal covariate whose effect is to be decomposed into direct
            and indirect components with respect to the rest of the right-hand side.
        x1_vars : list[str]
            A list of covariates that are included in both the baseline and the full
            regressions.
        decomp_var : str
            The name of the focal covariate whose effect is to be decomposed into direct
            and indirect components with respect to the rest of the right-hand side.
        type : str, optional
            The type of decomposition method to use. Defaults to "gelbach", which
            currently is the only supported option.
        cluster: Optional
            The name of the cluster variable. If None, uses the cluster variable
            from the model fit. Defaults to None.
        combine_covariates: Optional.
            A dictionary that specifies which covariates to combine into groups.
            See the example for how to use this argument. Defaults to None.
        reps : int, optional
            The number of bootstrap iterations to run. Defaults to 1000.
        seed : int, optional
            An integer to set the random seed. Defaults to None.
        nthreads : int, optional
            The number of threads to use for the bootstrap. Defaults to None.
            If None, uses all available threads minus one.
        agg_first : bool, optional
            If True, use the 'aggregate first' algorithm described in Gelbach (2016).
            False by default, unless combine_covariates is provided.
            Recommended to set to True if combine_covariates is argument is provided.
            As a rule of thumb, the more covariates are combined, the larger the performance
            improvement.
        only_coef : bool, optional
            Indicates whether to compute inference for the decomposition. Defaults to False.
            If True, skips the inference step and only returns the decomposition results.
        digits : int, optional
            The number of digits to round the results to. Defaults to 4.

        Returns
        -------
        GelbachDecomposition
            A GelbachDecomposition object with the decomposition results.
            Use `tidy()` and `etable()` to access the estimation results.

        Examples
        --------
        ```{python}
        import re
        import pyfixest as pf
        from pyfixest.utils.dgps import gelbach_data

        data = gelbach_data(nobs = 1000)
        fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

        # simple decomposition
        gb = fit.decompose(decomp_var = "x1", reps = 10, nthreads = 1)
        type(gb)

        gb.tidy()
        gb = fit.decompose(decomp_var = "x1", reps = 10, nthreads = 1, x1_vars = ["x21"])
        # combine covariates
        gb = fit.decompose(decomp_var = "x1", reps = 10, nthreads = 1, combine_covariates = {"g1": ["x21", "x22"], "g2": ["x23"]})
        # supress inference
        gb = fit.decompose(decomp_var = "x1", reps = 10, nthreads = 1, combine_covariates = {"g1": ["x21", "x22"], "g2": ["x23"]}, only_coef = True)
        # print results
        gb.etable()

        # group covariates via regex
        res = fit.decompose(decomp_var="x1", combine_covariates={"g1": re.compile("x2[1-2]"), "g2": re.compile("x23")})
        ```
        """
        has_param = param is not None
        has_decomp = decomp_var is not None

        if not has_param and not has_decomp:
            raise ValueError("Either 'param' or 'decomp_var' must be provided.")

        if has_param and has_decomp:
            raise ValueError(
                "The 'param' and 'decomp_var' arguments cannot be provided at the same time."
            )

        if has_param:
            warnings.warn(
                "The 'param' argument is deprecated. Please use 'decomp_var' instead.",
                UserWarning,
            )
            decomp_var = param

        if x1_vars is not None:
            if isinstance(x1_vars, str):
                x1_vars = [x.strip() for x in x1_vars.split("+")]
            else:
                x1_vars = list(x1_vars)

        self._require_support("decompose", subject="Decomposition")
        _decompose_arg_check(
            type=type,
            has_weights=self._has_weights,
            only_coef=only_coef,
        )

        self._require_fit_arrays("decompose", arrays="the fitted arrays")
        # A cluster variable or an absorbed fixed effect is read back from the
        # estimation sample; the plain covariate case works on arrays alone.
        if cluster is not None or self._is_clustered or self._has_fixef:
            self._require_estimation_data("decompose")

        nthreads_int = -1 if nthreads is None else nthreads

        rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

        if agg_first is None:
            agg_first = combine_covariates is not None

        cluster_df: pd.Series | None = None
        if cluster is not None:
            cluster_df = self._data[cluster]
        elif self._is_clustered:
            cluster_df = self._data[self._clustervar[0]]
        else:
            cluster_df = None

        Y, X, xnames = self._model_matrix_one_hot(output="sparse")

        if combine_covariates is not None:
            for key, value in combine_covariates.items():
                if isinstance(value, re.Pattern):
                    matched = [x for x in xnames if value.search(x)]
                    if len(matched) == 0:
                        raise ValueError(f"No covariates match the regex {value}.")
                    combine_covariates[key] = matched

        med = GelbachDecomposition(
            decomp_var=cast(str, decomp_var),
            x1_vars=x1_vars,
            coefnames=xnames,
            depvarname=self._depvar,
            cluster_df=cluster_df,
            nthreads=nthreads_int,
            combine_covariates=combine_covariates,
            agg_first=agg_first,
            only_coef=only_coef,
            atol=1e-12,
            btol=1e-12,
        )

        med.fit(
            X=X,
            Y=Y,
            weights=self._observation_weights.values,
            store=True,
        )

        if not only_coef:
            med.bootstrap(rng=rng, B=reps)

        self.GelbachDecompositionResults = med

        return med

    def update(
        self, X_new: np.ndarray, y_new: np.ndarray, inplace: bool = False
    ) -> np.ndarray:
        """
        Update coefficients for new observations using Sherman-Morrison formula.

        Parameters
        ----------
        X_new : np.ndarray
            Covariates for new data points. Users expected to ensure conformability
            with existing data.
        y_new : np.ndarray
            Outcome values for new data points.
        inplace : bool, optional
            Must be `False`. In-place updates are unsupported because appending
            design rows cannot reconstruct the complete fitted-result state.

        Returns
        -------
        np.ndarray
            Updated coefficients.

        Notes
        -----
        Updates the coefficients in closed form via the Sherman-Morrison
        identity instead of refitting on the full sample. `X_new` has to include
        the intercept column. The returned coefficients do not mutate the fitted
        result. Models with fixed effects are not supported.

        Examples
        --------
        Fit on all but the last observation, then add it:

        ```{python}
        import numpy as np
        import pyfixest as pf

        data = pf.get_data().dropna()
        fit = pf.feols("Y ~ X1 + X2", data.iloc[:-1])

        last = data.iloc[[-1]]
        X_new = np.column_stack(
            [np.ones(1), last["X1"].to_numpy(), last["X2"].to_numpy()]
        )
        y_new = last["Y"].to_numpy()

        fit.update(X_new, y_new)
        ```
        """
        self._require_support("update", subject="The update() method")
        if inplace:
            raise NotImplementedError(
                "update(..., inplace=True) is not supported because appending design "
                "rows cannot safely update the complete fitted-result state; use the "
                "returned coefficients instead."
            )
        self._require_fit_arrays("update", arrays="the fitted design arrays")
        if not np.all(X_new[:, 0] == 1):
            X_new = np.column_stack((np.ones(len(X_new)), X_new))
        X_n_plus_1 = np.vstack((self._X, X_new))
        epsi_n_plus_1 = y_new - X_new @ self._beta_hat
        gamma_n_plus_1 = np.linalg.inv(X_n_plus_1.T @ X_n_plus_1) @ X_new.T
        beta_n_plus_1 = self._beta_hat + gamma_n_plus_1 @ epsi_n_plus_1

        return beta_n_plus_1

    def evalue(
        self,
        mixture_precision: float = 1.0,
    ) -> pd.Series:
        """Compute coefficient-wise SAVI e-values.

        Parameters
        ----------
        mixture_precision : float, optional
            Positive mixture precision fixed before sequential monitoring.
            Defaults to 1. Use `pyfixest.optimal_mixture_precision()` to
            minimize confidence-sequence width at a target sample size.

        Returns
        -------
        pd.Series
            One e-value per coefficient.

        Notes
        -----
        SAVI currently supports unweighted, non-IV `feols` models without
        absorbed fixed effects. The covariance estimator must be iid or
        heteroskedasticity robust (`hetero`, `HC1`, `HC2`, or `HC3`). Note that
        for `HC2`/`HC3`, pyfixest's default small-sample correction scales the
        variance by `n / (n - k)` while the R implementation in `avlm` does not.
        Inference is pointwise / by coefficient.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2", data=data, vcov="hetero")
        fit.evalue()
        ```
        """
        from pyfixest.estimation.post_estimation.savi import _evalue

        return _evalue(model=self, mixture_precision=mixture_precision)

    def pvalue_savi(
        self,
        mixture_precision: float = 1.0,
    ) -> pd.Series:
        """Compute coefficient-wise SAVI sequential p-values.

        The sequential-p-value analogue of `evalue`. See `evalue` for the
        `mixture_precision` argument and the supported-model restrictions.

        Returns
        -------
        pd.Series
            One sequential p-value per coefficient.

        Examples
        --------
        ```{python}
        import pyfixest as pf

        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2", data=data, vcov="HC1")
        fit.pvalue_savi()
        ```
        """
        from pyfixest.estimation.post_estimation.savi import _pvalue_savi

        return _pvalue_savi(model=self, mixture_precision=mixture_precision)
