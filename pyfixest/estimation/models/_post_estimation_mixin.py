from importlib import import_module
import re
import warnings
from typing import Literal, Optional, Union, cast

import numpy as np
import pandas as pd
from formulaic import Formula
from scipy.sparse import csc_matrix, spmatrix
from scipy.stats import t

from pyfixest.estimation.post_estimation.decomposition import (
    GelbachDecomposition,
    _decompose_arg_check,
)
from pyfixest.estimation.post_estimation.ritest import (
    _decode_resampvar,
    _get_ritest_pvalue,
    _get_ritest_stats_fast,
    _get_ritest_stats_slow,
    _plot_ritest_pvalue,
)

decomposition_type = Literal["gelbach"]


class PostEstimationMixin:
    def wildboottest(
        self,
        reps: int,
        cluster: Optional[str] = None,
        param: Optional[str] = None,
        weights_type: Optional[str] = "rademacher",
        impose_null: Optional[bool] = True,
        bootstrap_type: Optional[str] = "11",
        seed: Optional[int] = None,
        k_adj: Optional[bool] = True,
        G_adj: Optional[bool] = True,
        parallel: Optional[bool] = False,
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

        if not self._supports_wildboottest:
            if self._is_iv:
                raise NotImplementedError(
                    "Wild cluster bootstrap is not supported for IV estimation."
                )
            if self._has_weights:
                raise NotImplementedError(
                    "Wild cluster bootstrap is not supported for WLS estimation."
                )

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

        if self._is_iv:
            raise NotImplementedError(
                "Wild cluster bootstrap is not supported with IV estimation."
            )

        if self._method == "fepois":
            raise NotImplementedError(
                "Wild cluster bootstrap is not supported for Poisson regression."
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
            "ssc": boot.small_sample_correction if run_heteroskedastic else boot.ssc,
        }

        res_df = pd.Series(res)

        if return_bootstrapped_t_stats:
            return res_df, boot.t_boot
        else:
            return res_df

    def ccv(
        self,
        treatment,
        cluster: Optional[str] = None,
        seed: Optional[int] = None,
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
        assert self._supports_cluster_causal_variance, (
            "The model does not support the causal cluster variance estimator."
        )
        assert isinstance(treatment, str), "treatment must be a string."
        assert isinstance(cluster, str) or cluster is None, (
            "cluster must be a string or None."
        )
        assert isinstance(seed, int) or seed is None, "seed must be an integer or None."
        assert isinstance(n_splits, int), "n_splits must be an integer."
        assert isinstance(pk, (int, float)) and 0 <= pk <= 1
        assert isinstance(qk, (int, float)) and 0 <= qk <= 1

        if self._has_fixef:
            raise NotImplementedError(
                "The causal cluster variance estimator is currently not supported for models with fixed effects."
            )

        if treatment not in self._coefnames:
            raise ValueError(
                f"Variable {treatment} not found in the model's coefficients."
            )

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

        depvar = self._depvar
        fml = self._fml
        xfml_list = fml.split("~")[1].split("+")
        xfml_list = [x for x in xfml_list if x != treatment]
        xfml = "" if not xfml_list else "+".join(xfml_list)

        data = self._data
        Y = self._Y.flatten()
        W = data[treatment].to_numpy()
        assert np.all(np.isin(W, [0, 1])), (
            "Treatment variable must be binary with values 0 and 1"
        )
        X = self._X
        cluster_vec = data[cluster].to_numpy()
        unique_clusters = np.unique(cluster_vec)

        tau_full = np.array(self.coef().xs(treatment))

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

        res_ccv_dict: dict[str, Union[float, np.ndarray]] = {
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

        ccv_module = import_module("pyfixest.estimation.post_estimation.ccv")
        _ccv = ccv_module._ccv

        return _ccv(
            data=data,
            depvar=depvar,
            treatment=treatment,
            cluster=cluster,
            xfml=xfml,
            seed=seed,
            pk=pk,
            qk=qk,
            n_splits=n_splits,
        )

    def _model_matrix_one_hot(
        self, output="numpy"
    ) -> tuple[np.ndarray, Union[np.ndarray, spmatrix], list[str]]:
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

            Y, X = Formula(fml_dummies).get_model_matrix(self._data, output=output)
            xnames = X.model_spec.column_names
            Y = Y.toarray().flatten() if output == "sparse" else Y.flatten()
            X = csc_matrix(X) if output == "sparse" else X

        else:
            Y = self._Y.flatten() / np.sqrt(self._weights.flatten())
            X = self._X / np.sqrt(self._weights)
            xnames = self._coefnames

        X = csc_matrix(X) if output == "sparse" else X

        return Y, X, xnames

    def decompose(
        self,
        param: Optional[str] = None,
        x1_vars: Optional[Union[list[str], str]] = None,
        decomp_var: Optional[str] = None,
        type: decomposition_type = "gelbach",
        cluster: Optional[str] = None,
        combine_covariates: Optional[dict[str, list[str]]] = None,
        reps: int = 1000,
        seed: Optional[int] = None,
        nthreads: Optional[int] = None,
        agg_first: Optional[bool] = None,
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

        _decompose_arg_check(
            type=type,
            has_weights=self._has_weights,
            weights_type=self._weights_type,
            is_iv=self._is_iv,
            method=self._method,
            only_coef=only_coef,
        )

        nthreads_int = -1 if nthreads is None else nthreads

        rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

        if agg_first is None:
            agg_first = combine_covariates is not None

        cluster_df: Optional[pd.Series] = None
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
            weights=self._weights,
            store=True,
        )

        if not only_coef:
            med.bootstrap(rng=rng, B=reps)

        self.GelbachDecompositionResults = med

        return med

    def ritest(
        self,
        resampvar: str,
        cluster: Optional[str] = None,
        reps: int = 100,
        type: str = "randomization-c",
        rng: Optional[np.random.Generator] = None,
        choose_algorithm: str = "auto",
        store_ritest_statistics: bool = False,
        level: float = 0.95,
    ) -> pd.Series:
        """
        Conduct Randomization Inference (RI) test against a null hypothesis of
        `resampvar = 0`.

        Parameters
        ----------
        resampvar : str
            The name of the variable to be resampled.
        cluster : str, optional
            The name of the cluster variable in case of cluster random assignment.
            If provided, `resampvar` is held constant within each `cluster`.
            Defaults to None.
        reps : int, optional
            The number of randomization iterations. Defaults to 100.
        type: str
            The type of the randomization inference test.
            Can be "randomization-c" or "randomization-t". Note that
            the "randomization-c" is much faster, while the
            "randomization-t" is recommended by Wu & Ding (JASA, 2021).
        rng : np.random.Generator, optional
            A random number generator. Defaults to None.
        choose_algorithm: str, optional
            The algorithm to use for the computation. Defaults to "auto".
            The alternative is "fast" and "slow", and should only be used
            for running CI tests. Ironically, this argument is not tested
            for any input errors from the user! So please don't use it =)
        include_plot: bool, optional
            Whether to include a plot of the distribution p-values. Defaults to False.
        store_ritest_statistics: bool, optional
            Whether to store the simulated statistics of the RI procedure.
            Defaults to False. If True, stores the simulated statistics
            in the model object via the `ritest_statistics` attribute as a
            numpy array.
        level: float, optional
            The level for the confidence interval of the randomization inference
            p-value. Defaults to 0.95.

        Returns
        -------
        A pd.Series with the regression coefficient of `resampvar` and the p-value
        of the RI test. Additionally, reports the standard error and the confidence
        interval of the p-value.

        Examples
        --------
        ```{python}

        #| echo: true
        #| results: asis
        #| include: true

        import pyfixest as pf
        data = pf.get_data()
        fit = pf.feols("Y ~ X1 + X2", data=data)

        # Conduct a randomization inference test for the coefficient of X1
        fit.ritest("X1", reps=1000)

        # use randomization-t instead of randomization-c
        fit.ritest("X1", reps=1000, type="randomization-t")

        # store statistics for plotting
        fit.ritest("X1", reps=1000, store_ritest_statistics=True)
        ```
        """
        resampvar = resampvar.replace(" ", "")
        resampvar_, h0_value, hypothesis, test_type = _decode_resampvar(resampvar)

        if self._is_iv:
            raise NotImplementedError(
                "Randomization Inference is not supported for IV models."
            )

        # check that resampvar in _coefnames
        if resampvar_ not in self._coefnames:
            raise ValueError(f"{resampvar_} not found in the model's coefficients.")

        if cluster is not None and cluster not in self._data:
            raise ValueError(f"The variable {cluster} is not found in the data.")

        clustervar_arr = (
            self._data[cluster].to_numpy().reshape(-1, 1) if cluster else None
        )

        if clustervar_arr is not None and np.any(np.isnan(clustervar_arr)):
            raise ValueError(
                """
            The cluster variable contains missing values. This is not allowed
            for randomization inference via `ritest()`.
            """
            )

        # update vcov if cluster provided but not in model
        if cluster is not None and not self._is_clustered:
            warnings.warn(
                "The initial model was not clustered. CRV1 inference is computed and stored in the model object."
            )
            self.vcov({"CRV1": cluster})

        rng = np.random.default_rng() if rng is None else rng

        sample_coef = np.array(self.coef().xs(resampvar_))
        sample_tstat = np.array(self.tstat().xs(resampvar_))
        sample_stat = sample_tstat if type == "randomization-t" else sample_coef

        if type not in ["randomization-t", "randomization-c"]:
            raise ValueError("type must be 'randomization-t' or 'randomization-c.")

        # always run slow algorithm for randomization-t
        choose_algorithm = "slow" if type == "randomization-t" else choose_algorithm

        assert isinstance(reps, int) and reps > 0, "reps must be a positive integer."

        if self._has_weights:
            raise NotImplementedError(
                """
                Regression Weights are not supported with Randomization Inference.
                """
            )

        if choose_algorithm == "slow" or self._method == "fepois":
            vcov_input: Union[str, dict[str, str]]
            if cluster is not None:
                vcov_input = {"CRV1": cluster}
            else:
                # "iid" for models without controls, else HC1
                vcov_input = (
                    "hetero"
                    if (self._has_fixef and len(self._coefnames) > 1)
                    or len(self._coefnames) > 2
                    else "iid"
                )

            # for performance reasons
            if type == "randomization-c":
                vcov_input = "iid"

            ri_stats = _get_ritest_stats_slow(
                data=self._data,
                resampvar=resampvar_,
                clustervar_arr=clustervar_arr,
                fml=self._fml,
                reps=reps,
                vcov=vcov_input,
                type=type,
                rng=rng,
                model=self._method,
            )

        else:
            weights = self._weights.flatten()
            fval_df = (
                self._data[self._fixef.split("+")] if self._fixef is not None else None
            )
            D = self._data[resampvar_].to_numpy()

            ri_stats = _get_ritest_stats_fast(
                Y=self._Y,
                X=self._X,
                D=D,
                coefnames=self._coefnames,
                resampvar=resampvar_,
                clustervar_arr=clustervar_arr,
                reps=reps,
                rng=rng,
                fval_df=fval_df,
                weights=weights,
            )

        ri_pvalue, se_pvalue, ci_pvalue = _get_ritest_pvalue(
            sample_stat=sample_stat,
            ri_stats=ri_stats[1:],
            method=test_type,
            h0_value=h0_value,
            level=level,
        )

        if store_ritest_statistics:
            self._ritest_statistics = ri_stats
            self._ritest_pvalue = ri_pvalue
            self._ritest_sample_stat = sample_stat - h0_value

        res = pd.Series(
            {
                "H0": hypothesis,
                "ri-type": type,
                "Estimate": sample_coef,
                "Pr(>|t|)": ri_pvalue,
                "Std. Error (Pr(>|t|))": se_pvalue,
            }
        )

        alpha = 1 - level
        ci_lower_name = str(f"{alpha / 2 * 100:.1f}% (Pr(>|t|))")
        ci_upper_name = str(f"{(1 - alpha / 2) * 100:.1f}% (Pr(>|t|))")
        res[ci_lower_name] = ci_pvalue[0]
        res[ci_upper_name] = ci_pvalue[1]

        if cluster is not None:
            res["Cluster"] = cluster

        return res

    def plot_ritest(self, plot_backend="lets_plot"):
        """
        Plot the distribution of the Randomization Inference Statistics.

        Parameters
        ----------
        plot_backend : str, optional
            The plotting backend to use. Defaults to "lets_plot". Alternatively,
            "matplotlib" is available.

        Returns
        -------
        A lets_plot or matplotlib figure with the distribution of the Randomization
        Inference Statistics.
        """
        if not hasattr(self, "_ritest_statistics"):
            raise ValueError(
                """
                            The randomization inference statistics have not been stored
                            in the model object. Please set `store_ritest_statistics=True`
                            when calling `ritest()`
                            """
            )

        ri_stats = self._ritest_statistics
        sample_stat = self._ritest_sample_stat

        return _plot_ritest_pvalue(
            ri_stats=ri_stats, sample_stat=sample_stat, plot_backend=plot_backend
        )
