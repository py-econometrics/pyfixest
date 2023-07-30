import pyhdfe
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from typing import Any, Union, Dict, Optional, List, Tuple
from scipy.stats import norm
from formulaic import model_matrix

from pyfixest.feols import Feols
from pyfixest.FormulaParser import FixestFormulaParser, _flatten_list
from pyfixest.ssc_utils import ssc
from pyfixest.exceptions import MatrixNotFullRankError, MultiEstNotSupportedError


class Fixest:

    def __init__(self, data: pd.DataFrame) -> None:
        '''
        A class for fixed effects regression modeling.
        Args:
            data: The input pd.DataFrame for the object.
        Returns:
            None
        '''

        self.data = data.copy()
        self.all_fitted_models = dict()


    def feols(self, fml: str, vcov: Union[None, str, Dict[str, str]] = None, ssc=ssc(), fixef_rm: str = "none") -> None:
        '''
        Method for fixed effects regression modeling using the PyHDFE package for projecting out fixed effects.
        Args:
            fml (str): A three-sided formula string using fixest formula syntax. Supported syntax includes:
                The syntax is as follows: "Y ~ X1 + X2 | FE1 + FE2 | X1 ~ Z1" where:

                Y: Dependent variable
                X1, X2: Independent variables
                FE1, FE2: Fixed effects
                Z1, Z2: Instruments
                |: Separates left-hand side, fixed effects, and instruments

                If no fixed effects and instruments are specified, the formula can be simplified to "Y ~ X1 + X2".
                If no instruments are specified, the formula can be simplified to "Y ~ X1 + X2 | FE1 + FE2".
                If no fixed effects are specified but instruments are specified, the formula can be simplified to "Y ~ X1 + X2 | X1 ~ Z1".

                Supported multiple estimation syntax includes:

                Stepwise regressions (sw, sw0)
                Cumulative stepwise regression (csw, csw0)
                Multiple dependent variables (Y1 + Y2 ~ X)

                Other special syntax includes:
                i() for interaction of a categorical and non-categorical variable (e.g. "i(X1,X2)" for interaction between X1 and X2).
                Using i() is required to use with some custom methods, e.g. iplot().
                ^ for interacted fixed effects (e.g. "fe1^fe2" for interaction between fe1 and fe2)

                All other parts of the formula must be compatible with formula parsing via the formulaic module.
                You can use formulaic functionaloty such as "C", "I", ":",, "*", "np.log", "np.power", etc.

            vcov (Union(str, dict)): A string or dictionary specifying the type of variance-covariance matrix to use for inference.
                If a string, it can be one of "iid", "hetero", "HC1", "HC2", "HC3".
                If a dictionary, it should have the format dict("CRV1":"clustervar") for CRV1 inference or dict(CRV3":"clustervar") for CRV3 inference.
            fixef_rm: A string specifiny whether singleton fixed effects should be dropped. Options are "none" (default) and "singleton". If "singleton", singleton fixed effects are dropped.
        Returns:
            None
        Examples:
            Standard formula:
                fml = 'Y ~ X1 + X2'
                fixest_model = Fixest(data=data).feols(fml, vcov='iid')
            With fixed effects:
                fml = 'Y ~ X1 + X2 | fe1 + fe2'
            With interacted fixed effects:
                fml = 'Y ~ X1 + X2 | fe1^fe2'
            Multiple dependent variables:
                fml = 'Y1 + Y2 ~ X1 + X2'
            Stepwise regressions (sw and sw0):
                fml = 'Y1 + Y2 ~ sw(X1, X2, X3)'
            Cumulative stepwise regressions (csw and csw0):
                fml = 'Y1 + Y2 ~ csw(X1, X2, X3) '
            Combinations:
                fml = 'Y1 + Y2 ~ csw(X1, X2, X3) | sw(X4, X5) + X6'

        Details:
            The method proceeds in the following steps:
            1. Parse the formula using the FixestFormulaParser class.
            2. Create a dictionary of formulas for each dependent variable.
            3. demean all models and store the data
            4. fit all models
        '''

        self.fml = fml.replace(" ", "")
        self.split = None
        self.method = "feols"

        # deparse formula, at least partially
        fxst_fml = FixestFormulaParser(fml)

        if fxst_fml.is_iv:
            self.is_iv = True
        else:
            self.is_iv = False

        # add function argument to these methods for IV
        fxst_fml.get_new_fml_dict() # fxst_fml.fml_dict_new

        self.fml_dict = fxst_fml.fml_dict_new

        if self.is_iv:
            fxst_fml.get_new_fml_dict(iv = True) # fxst_fml.fml_dict_new
            self.fml_dict_iv = fxst_fml.fml_dict_new_iv
        else:
            self.fml_dict_iv = None

        self.ivars = fxst_fml.ivars

        self.ssc_dict = ssc
        self.drop_singletons = _drop_singletons(fixef_rm)

        # get all fixed effects combinations
        fixef_keys = list(self.fml_dict.keys())

        self.ivars, self.drop_ref = _clean_ivars(self.ivars, self.data)

        # names of depvar, X, Z matrices
        self.yxz_name_dict = dict()

        # currently no fsplit allowed
        fsplit = None

        self.splitvar, _, self.estimate_split_model, self.estimate_full_model = _prepare_split_estimation(self.split, fsplit, self.data, self.fml_dict)

        # demean all models: based on fixed effects x split x missing value combinations
        self._estimate_all_models2(vcov, fixef_keys)

        # create self.is_fixef_multi flag
        self._is_multiple_estimation()

        if self.is_fixef_multi and self.is_iv:
            raise MultiEstNotSupportedError(
                "Multiple Estimations is currently not supported with IV."
                "This is mostly due to insufficient testing and will be possible with the next release of PyFixest."
            )

        return self


    def _clean_fe(self, data, fval):

        '''
        Function that transform and cleans fixed effects.
        '''

        fval_list = fval.split("+")

        # find interacted fixed effects via "^"
        interacted_fes = [x for x in fval_list if len(x.split('^')) > 1]

        varying_slopes = [x for x in fval_list if len(x.split('/')) > 1]

        for x in interacted_fes:
            vars = x.split("^")
            data[x] = data[vars].apply(lambda x: '^'.join(
                x.dropna().astype(str)) if x.notna().all() else np.nan, axis=1)

        fe = data[fval_list]
        # all fes to factors / categories

        if varying_slopes != []:

            for x in varying_slopes:
                mm_vs = model_matrix("-1 + " + x, data)

            fe = pd.concat([fe, mm_vs], axis = 1)

        fe_na = fe.isna().any(axis=1)
        fe = fe.apply(lambda x: pd.factorize(x)[0])
        fe = fe.to_numpy()

        return fe, fe_na


    def _model_matrix_fixest(self, depvar, covar, fval):

        '''
        Create model matrices for fixed effects estimation. Preprocesses the data and then calls
        formulaic.model_matrix() to create the model matrices.

        Args:
            depvar: dependent variable. string. E.g. "Y"
            covar: covariates. string. E.g. "X1 + X2"
            fval: fixed effects. string. E.g. "fe1 + fe2". "0" if no fixed effects.
        Returns:
            Y: a pd.DataFrame of the dependent variable.
            X: a pd.DataFrame of the covariates
            I: a pd.DataFrame of the Instruments. None if no IV.
            fe: a pd.DataFrame of the fixed effects. None if no fixed effects specified.
            na_index: a np.array with indices of dropped columns.
            fe_na: a np.array with indices of dropped columns due to fixed effect singletons / NaNs in the fixed effects
            na_index_str: na_index, but as a comma separated string. Used for caching of demeaned variables
            z_names: names of all covariates, minus the endogeneous variables, plus the instruments. None if no IV.
        '''

        if fval != "0":
            fe, fe_na = self._clean_fe(self.data, fval)
            fe_na = list(fe_na[fe_na == True])
            fe = pd.DataFrame(fe)
        else:
            fe = None
            fe_na = None

        if self.is_iv:
            dict2fe_iv = self.fml_dict_iv.get(fval)

        covar2 = covar
        depvar2 = depvar

        fml = depvar2 + " ~ " + covar2

        if self.is_iv:
            instruments2 = dict2fe_iv.get("Y")[0].split("~")[1]
            endogvar_list = list(set(covar2.split("+")) - set(instruments2.split("+")))#[0]
            instrument_list = list(set(instruments2.split("+")) - set(covar2.split("+")))#[0]

            fml2 = "+".join(instrument_list) + "+" + fml

        else:
            fml2 = fml

        lhs, rhs = model_matrix(fml2, self.data)

        Y = lhs[[depvar]]
        X = pd.DataFrame(rhs)
        if self.is_iv:
            I = lhs[instrument_list]
            I = pd.DataFrame(I)
        else:
            I = None

        # get NA index before converting Y to numpy array
        na_index = list(set(self.data.index) - set(Y.index))

        # drop variables before collecting variable names
        if self.ivars is not None:
            if self.drop_ref is not None:
                X = X.drop(self.drop_ref, axis=1)

        y_names = list(Y.columns)
        x_names = list(X.columns)
        yxz_names = list(y_names) + list(x_names)

        if self.is_iv:
            iv_names = list(I.columns)
            x_names_copy = x_names.copy()
            x_names_copy = [x for x in x_names_copy if x not in endogvar_list]
            z_names = x_names_copy + instrument_list
            cols = yxz_names + iv_names
        else:
            iv_names = None
            z_names = None
            cols = yxz_names

        if self.ivars is not None:
            self.icovars = [s for s in x_names if s.startswith(
                self.ivars[0]) and s.endswith(self.ivars[1])]
        else:
            self.icovars = None

        if Y.shape[1] > 1:
            raise ValueError(
                "Dependent variable must be a single column."
                "Please make sure that the dependent variable" + depvar2 + "is of a numeric type (int or float)."
           )

        if fe is not None:
            na_index = (na_index + fe_na)
            fe = fe.drop(na_index, axis=0)
            # drop intercept
            X = X.drop('Intercept', axis = 1)
            x_names.remove("Intercept")
            yxz_names.remove("Intercept")
            if self.is_iv:
                z_names.remove("Intercept")
                cols.remove("Intercept")

            # check if variables have already been demeaned
            Y = Y.drop(fe_na, axis=0)
            X = X.drop(fe_na, axis=0)
            if self.is_iv:
                I = I.drop(fe_na, axis=0)

        na_index_str = ','.join(str(x) for x in na_index)

        return Y, X, I, fe, na_index, fe_na, na_index_str, z_names


    def _demean_model2(self, Y, X, I, fe, lookup_demeaned_data, na_index_str):

        '''
        Demeans a single regression model. If the model has fixed effects, the fixed effects are demeaned using the PyHDFE package.
        Prior to demeaning, the function checks if some of the variables have already been demeaned and uses values from the cache
        `lookup_demeaned_data` if possible. If the model has no fixed effects, the function does not demean the data.

        Args:
            Y: a pd.DataFrame of the dependent variable.
            X: a pd.DataFrame of the covariates
            I: a pd.DataFrame of the Instruments. None if no IV.
            fe: a pd.DataFrame of the fixed effects. None if no fixed effects specified.
            lookup_demeaned_data: a dictionary with keys for each fixed effects combination and potentially values of demeaned data.frames.
                The function checks this dictionary to see if some of the variables have already been demeaned.
            na_index_str: a string with indices of dropped columns. Used for caching of demeaned variables.

        Returns:
            Yd: a pd.DataFrame of the demeaned dependent variable.
            Xd: a pd.DataFrame of the demeaned covariates
            Id: a pd.DataFrame of the demeaned Instruments. None if no IV.
        '''

        if I is not None:
            YXZ = pd.concat([Y, X, I], axis = 1)
        else:
            YXZ = pd.concat([Y, X], axis = 1)

        yxz_names = YXZ.columns
        YXZ = YXZ.to_numpy()

        if fe is not None:

            # check if looked dict has data for na_index
            if lookup_demeaned_data.get(na_index_str) is not None:
                # get data out of lookup table: list of [algo, data]
                algorithm, YXZ_demeaned_old = lookup_demeaned_data.get(na_index_str)

                # get not yet demeaned covariates
                var_diff_names = list(set(yxz_names) - set(YXZ_demeaned_old.columns))[0]
                var_diff_index = list(yxz_names).index(var_diff_names)
                var_diff = YXZ[:, var_diff_index]
                if var_diff.ndim == 1:
                    var_diff = var_diff.reshape(len(var_diff), 1)

                YXZ_demean_new = algorithm.residualize(var_diff)
                YXZ_demeaned = np.concatenate([YXZ_demeaned_old, YXZ_demean_new], axis=1)
                YXZ_demeaned = pd.DataFrame(YXZ_demeaned)

                YXZ_demeaned.columns = list(YXZ_demeaned_old.columns) + [var_diff_names]

            else:

                # not data demeaned yet for NA combination
                algorithm = pyhdfe.create(
                    ids=fe,
                    residualize_method='map',
                    drop_singletons=self.drop_singletons,
                )

                if self.drop_singletons == True and algorithm.singletons != 0 and algorithm.singletons is not None:
                    print(algorithm.singletons, "columns are dropped due to singleton fixed effects.")
                    dropped_singleton_indices = np.where(algorithm._singleton_indices)[0].tolist()
                    na_index += dropped_singleton_indices

                YXZ_demeaned = algorithm.residualize(YXZ)
                YXZ_demeaned = pd.DataFrame(YXZ_demeaned)

                YXZ_demeaned.columns = yxz_names

            lookup_demeaned_data[na_index_str] = [algorithm, YXZ_demeaned]

        else:
            # nothing to demean here
            pass

            YXZ_demeaned = pd.DataFrame(YXZ)
            YXZ_demeaned.columns = yxz_names

        # get demeaned Y, X, I (if no fixef, equal to Y, X, I)
        Yd = YXZ_demeaned[Y.columns]
        Xd = YXZ_demeaned[X.columns]
        Id = None
        if I is not None:
            Id = YXZ_demeaned[I.columns]


        return Yd, Xd, Id



    def _estimate_all_models2(self, vcov, fixef_keys):

        '''
        demean multiple models. essentially, the function loops
        over all split var and fixed effects variables and demeans the
        specified dependend variables and covariates
        Args:
            fixef_keys: fixed effect variables
            ivars: interaction variables
            drop_ref: drop reference category
            estimate_full_model: boolean, whether to estimate the full model
            estimate_split_model: boolean, whether to estimate the split model
        '''


        if self.estimate_full_model:

            for _, fval in enumerate(fixef_keys):

                dict2fe = self.fml_dict.get(fval)

                # dictionary to cache demeaned data with index: na_index_str
                lookup_demeaned_data = dict()

                # loop over both dictfe and dictfe_iv (if the latter is not None)
                for depvar in dict2fe.keys():

                    for _, fml_linear in enumerate(dict2fe.get(depvar)):

                        if self.method == "feols":

                            if isinstance(fml_linear, list):
                                fml_linear = fml_linear[0]

                            covar = fml_linear.split("~")[1]

                            # get Y, X, Z, fe, NA indices for model
                            Y, X, I, fe, na_index, _, na_index_str, z_names = self._model_matrix_fixest(depvar, covar, fval)

                            y_names = Y.columns.tolist()
                            x_names = X.columns.tolist()

                            fml = get_fml(y_names, x_names, fval)

                            # demean Y, X, Z, if not already done in previous estimation
                            Yd, Xd, Id = self._demean_model2(Y, X, I, fe, lookup_demeaned_data, na_index_str)
                            if self.is_iv:
                                Zd = pd.concat([Xd, Id], axis = 1)[z_names]
                            else:
                                Zd = Xd

                            Yd = Yd.to_numpy()
                            Xd = Xd.to_numpy()
                            Zd = Zd.to_numpy()

                            # check for multicollinearity
                            _multicollinearity_checks(Xd, Zd, self.ivars, fml)

                            # initiate OLS class
                            FEOLS = Feols(Y = Yd, X = Xd, Z = Zd)

                            # estimation
                            if self.is_iv:
                                FEOLS.get_fit(estimator = "2sls")
                                FEOLS.is_iv = True
                            else:
                                FEOLS.get_fit(estimator = "ols")
                                FEOLS.is_iv = False

                            # some bookkeeping
                            FEOLS.fml = fml
                            FEOLS.ssc_dict = self.ssc_dict
                            FEOLS.na_index = na_index
                            # data never makes it to Feols() class. needed for ex post
                            # clustered vcov estimation when clustervar not in model params
                            FEOLS.data = self.data.iloc[~self.data.index.isin(na_index)]
                            if fval != "0":
                                FEOLS.has_fixef = True
                                FEOLS.fixef = fval
                            else:
                                FEOLS.has_fixef = False
                                FEOLS.fixef = None
                            #FEOLS.split_log = x


                            # inference
                            vcov_type = _get_vcov_type(vcov, fval)

                            FEOLS.vcov_log = vcov_type
                            FEOLS.get_vcov(vcov=vcov_type)
                            FEOLS.get_inference()
                            FEOLS.coefnames = x_names
                            if self.icovars is not None:
                                FEOLS.icovars = self.icovars
                            else:
                                FEOLS.icovars = None

                            self.all_fitted_models[fml] = FEOLS

                        elif self.method == "fepois":

                          # estimation via FEPOIS
                          pass

                        else:

                            raise ValueError(
                                "Estimation method not supported. Please use 'feols' or 'fepois'."
                            )




    def _is_multiple_estimation(self):

        '''
        helper method to check if multiple regression models will be estimated
        '''

        self.is_fixef_multi = False
        if len(self.fml_dict.keys()) > 1:
            self.is_fixef_multi = True
        elif len(self.fml_dict.keys()) == 1:
            first_key = next(iter(self.fml_dict))
            if len(self.fml_dict[first_key]) > 1:
                self.is_fixef_multi = True


    def vcov(self, vcov: Union[str, Dict[str, str]]) -> None:
        '''
        Update regression inference "on the fly".
        By calling vcov() on a "Fixest" object, all inference procedures applied
        to the "Fixest" object are replaced with the variance covariance matrix specified via the method.
        Args:
            vcov: A string or dictionary specifying the type of variance-covariance matrix to use for inference.
                If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3".
                If a dictionary, it should have the format {"CRV1":"clustervar"} for CRV1 inference
                or {"CRV3":"clustervar"} for CRV3 inference.
        Returns:
            None
        '''

        self.vcov_log = vcov

        for model in list(self.all_fitted_models.keys()):

            fxst = self.all_fitted_models[model]
            fxst.vcov_log = vcov

            fxst.get_vcov(vcov=vcov)
            fxst.get_inference()

        return self

    def tidy(self, type: Optional[str] = None) -> Union[pd.DataFrame, str]:
        '''
        Returns the results of an estimation using `feols()` as a tidy Pandas DataFrame.
        Args:
            type : str, optional
                The type of output format to use. If set to "markdown", the resulting DataFrame
                will be returned in a markdown format with three decimal places. Default is None.
        Returns:
            pd.DataFrame or str
                A tidy DataFrame with the following columns:
                - fml: the formula used to generate the results
                - coefnames: the names of the coefficients
                - Estimate: the estimated coefficients
                - Std. Error: the standard errors of the estimated coefficients
                - t value: the t-values of the estimated coefficients
                - Pr(>|t|): the p-values of the estimated coefficients
                If `type` is set to "markdown", the resulting DataFrame will be returned as a
                markdown-formatted string with three decimal places.
        '''

        res = []
        for x in list(self.all_fitted_models.keys()):

            fxst = self.all_fitted_models[x]
            df = fxst.tidy().reset_index()
            df["fml"] = fxst.fml
            res.append(df)


        res = pd.concat(res, axis=0).set_index(['fml', 'coefnames'])
        if type == "markdown":
            return res.to_markdown(floatfmt=".3f")
        else:
            return res

    def summary(self) -> None:
        '''
        Prints a summary of the feols() estimation results for each estimated model.
        For each model, the method prints a header indicating the fixed-effects and the
        dependent variable, followed by a table of coefficient estimates with standard
        errors, t-values, and p-values.
        Returns:
            None
        '''

        for x in list(self.all_fitted_models.keys()):

            split = x.split("|")
            if len(split) > 1:
                fe = split[1]
            else:
                fe = None
            depvar = split[0].split("~")[0]
            fxst = self.all_fitted_models[x]

            df = fxst.tidy()

            if fxst.is_iv:
                estimation_method = "IV"
            else:
                estimation_method = "OLS"

            print('###')
            print('')
            print('Model: ', estimation_method)
            print('Dep. var.: ', depvar)
            if fe is not None:
                print('Fixed effects: ', fe)
            # if fxst.split_log is not None:
            #    print('Split. var: ', self.split + ":" + fxst.split_log)
            print('Inference: ', fxst.vcov_log)
            print('Observations: ', fxst.N)
            print('')
            print(df.to_string(index=False))
            print('---')

    def coef(self) -> pd.DataFrame:
        '''
        Obtain the coefficients of the fitted models.
        Returns:
            A pd.DataFrame with coefficient names and Estimates. The key indicates which models the estimated statistic derives from.
        '''
        return self.tidy()["Estimate"]

    def se(self)-> pd.DataFrame:
        '''
        Obtain the standard errors of the fitted models.

        Returns:
            A pd.DataFrame with coefficient names and standard error estimates. The key indicates which models the estimated statistic derives from.

        '''
        return self.tidy()["Std. Error"]


    def tstat(self)-> pd.DataFrame:
        '''
        Obtain the t-statistics of the fitted models.

         Returns:
            A pd.DataFrame with coefficient names and estimated t-statistics. The key indicates which models the estimated statistic derives from.

        '''
        return self.tidy()["t value"]

    def pvalue(self) -> pd.DataFrame:
        '''
        Obtain the p-values of the fitted models.

        Returns:
            A pd.DataFrame with coefficient names and p-values. The key indicates which models the estimated statistic derives from.

        '''
        return self.tidy()["Pr(>|t|)"]

    def confint(self) -> pd.DataFrame:
        '''
        Obtain the confidence intervals of the fitted models.
        Returns:
            A pd.DataFrame with coefficient names and confidence intervals. The key indicates which models the estimated statistic derives from.
        '''

        return self.tidy()[["confint_lower", "confint_upper"]]

    def iplot(self, alpha: float = 0.05, figsize: tuple = (10, 10), yintercept: Union[int, str, None] = None, xintercept: Union[int, str, None] = None, rotate_xticks: int = 0) -> None:
        '''
        Plot model coefficients with confidence intervals for variable interactions specified via the `i()` syntax.
        Args:
            alpha: float, optional. The significance level for the confidence intervals. Default is 0.05.
            figsize: tuple, optional. The size of the figure. Default is (10, 10).
            yintercept: int or str (for a categorical x axis). The value at which to draw a horizontal line.
            xintercept: int or str (for a categorical y axis). The value at which to draw a vertical line.
        Returns:
            None
        '''

        ivars = self.icovars

        if ivars is None:
            raise ValueError(
                "The estimated models did not have ivars / 'i()' model syntax."
                "In consequence, the '.iplot()' method is not supported."
            )

        if "Intercept" in ivars:
            ivars.remove("Intercept")

        df = self.tidy().reset_index()

        df = df[df.coefnames.isin(ivars)]
        models = df.fml.unique()

        _coefplot(
            models=models,
            figsize=figsize,
            alpha=alpha,
            yintercept=yintercept,
            xintercept=xintercept,
            df=df,
            is_iplot=True
        )

    def coefplot(self, alpha: float = 0.05, figsize: tuple = (5, 2), yintercept: int = 0, figtitle: str = None, figtext: str = None, rotate_xticks: int = 0) -> None:
        '''
        Plot estimation results. The plot() method is only defined for single regressions.
        Args:
            alpha (float): the significance level for the confidence intervals. Default is 0.05.
            figsize (tuple): the size of the figure. Default is (5, 2).
            yintercept (float): the value of the y-intercept. Default is 0.
            figtitle (str): the title of the figure. Default is None.
            figtext (str): the text at the bottom of the figure. Default is None.
        Returns:
            None
        '''

        df = self.tidy().reset_index()
        models = df.fml.unique()

        _coefplot(
            models=models,
            figsize=figsize,
            alpha=alpha,
            yintercept=yintercept,
            xintercept=None,
            df=df,
            is_iplot=False,
            rotate_xticks=rotate_xticks
        )

    def wildboottest(self, B, param: Union[str, None] = None, weights_type: str = 'rademacher', impose_null: bool = True, bootstrap_type: str = '11', seed: Union[str, None] = None, adj: bool = True, cluster_adj: bool = True) -> pd.DataFrame:

        '''
        Run a wild cluster bootstrap for all regressions in the Fixest object.

        Args:

            B (int): The number of bootstrap iterations to run
            param (Union[str, None], optional): A string of length one, containing the test parameter of interest. Defaults to None.
            weights_type (str, optional): The type of bootstrap weights. Either 'rademacher', 'mammen', 'webb' or 'normal'.
                                'rademacher' by default. Defaults to 'rademacher'.
            impose_null (bool, optional): Should the null hypothesis be imposed on the bootstrap dgp, or not?
                                Defaults to True.
            bootstrap_type (str, optional):A string of length one. Allows to choose the bootstrap type
                                to be run. Either '11', '31', '13' or '33'. '11' by default. Defaults to '11'.
            seed (Union[str, None], optional): Option to provide a random seed. Defaults to None.

        Returns:
            A pd.DataFrame with bootstrapped t-statistic and p-value. The index indicates which model the estimated statistic derives from.
        '''


        res = []
        for x in list(self.all_fitted_models.keys()):

            fxst = self.all_fitted_models[x]

            if hasattr(fxst, 'clustervar'):
                cluster = fxst.clustervar
            else:
                cluster = None

            boot_res = fxst.get_wildboottest(B, cluster, param,  weights_type, impose_null, bootstrap_type, seed, adj, cluster_adj)

            pvalue = boot_res["pvalue"]
            tstat = boot_res["statistic"]


            res.append(
                pd.Series(
                    {
                        'fml': x,
                        'param':param,
                        't value': tstat,
                        'Pr(>|t|)': pvalue
                    }
                )
            )

        res = pd.concat(res, axis=1).T.set_index('fml')

        return res


    def fetch_model(self, i: Union[int, str]):

        '''
        Utility method to fetch a model of class Feols from the Fixest class.
        Args:
            i (int or str): The index of the model to fetch.
        Returns:
            A Feols object.
        '''

        if isinstance(i, str):
            i = int(i)

        keys = list(self.all_fitted_models.keys())
        if i >= len(keys):
            raise IndexError(f"Index {i} is larger than the number of fitted models.")
        key = keys[i]
        print("Model: ", key)
        model = self.all_fitted_models[key]
        return model

def _coefplot(models: List, df: pd.DataFrame, figsize: Tuple[int, int], alpha: float, yintercept: Optional[int] = None,
              xintercept: Optional[int] = None, is_iplot: bool = False,
              rotate_xticks: float = 0) -> None:
    """
        Plot model coefficients with confidence intervals.
        Args:
            models (list): A list of fitted models indices.
            figsize (tuple): The size of the figure.
            alpha (float): The significance level for the confidence intervals.
            yintercept (int or None): The value at which to draw a horizontal line on the plot.
            xintercept (int or None): The value at which to draw a vertical line on the plot.
            df (pandas.DataFrame): The dataframe containing the data used for the model fitting.
            is_iplot (bool): If True, plot variable interactions specified via the `i()` syntax.
            rotate_xticks (float): The angle in degrees to rotate the xticks labels. Default is 0 (no rotation).
        Returns:
        None
    """

    if len(models) > 1:

        fig, ax = plt.subplots(len(models), gridspec_kw={
                               'hspace': 0.5}, figsize=figsize)

        for x, model in enumerate(models):

            df_model = df.reset_index().set_index("fml").xs(model)
            coef = df_model["Estimate"]
            conf_l = coef - \
                df_model["Std. Error"] * norm.ppf(1 - alpha / 2)
            conf_u = coef + \
                df_model["Std. Error"] * norm.ppf(1 - alpha / 2)
            coefnames = df_model["coefnames"].values.tolist()

            # could be moved out of the for loop, as the same ivars for all
            # models.

            if is_iplot == True:
                fig.suptitle("iplot")
                coefnames = [(i) for string in coefnames for i in re.findall(
                    r'\[T\.([\d\.\-]+)\]', string)]

            ax[x].scatter(coefnames, coef, color="b", alpha=0.8)
            ax[x].scatter(coefnames, conf_u, color="b",
                          alpha=0.8, marker="_", s=100)
            ax[x].scatter(coefnames, conf_l, color="b",
                          alpha=0.8, marker="_", s=100)
            ax[x].vlines(coefnames, ymin=conf_l,
                         ymax=conf_u, color="b", alpha=0.8)
            if yintercept is not None:
                ax[x].axhline(yintercept, color='red',
                              linestyle='--', alpha=0.5)
            if xintercept is not None:
                ax[x].axvline(xintercept, color='red',
                              linestyle='--', alpha=0.5)
            ax[x].set_ylabel('Coefficients')
            ax[x].set_title(model)
            ax[x].tick_params(axis='x', rotation=rotate_xticks)

    else:

        fig, ax = plt.subplots(figsize=figsize)

        model = models[0]

        df_model = df.reset_index().set_index("fml").xs(model)

        coef = df_model["Estimate"].values
        conf_l = coef - df_model["Std. Error"].values * norm.ppf(1 - alpha / 2)
        conf_u = coef + df_model["Std. Error"].values * norm.ppf(1 - alpha / 2)
        coefnames = df_model["coefnames"].values.tolist()

        if is_iplot == True:
            fig.suptitle("iplot")
            coefnames = [(i) for string in coefnames for i in re.findall(
                r'\[T\.([\d\.\-]+)\]', string)]

        ax.scatter(coefnames, coef, color="b", alpha=0.8)
        ax.scatter(coefnames, conf_u, color="b", alpha=0.8, marker="_", s=100)
        ax.scatter(coefnames, conf_l, color="b", alpha=0.8, marker="_", s=100)
        ax.vlines(coefnames, ymin=conf_l, ymax=conf_u, color="b", alpha=0.8)
        if yintercept is not None:
            ax.axhline(yintercept, color='red', linestyle='--', alpha=0.5)
        if xintercept is not None:
            ax.axvline(xintercept, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel('Coefficients')
        ax.set_title(model)
        ax.tick_params(axis='x', rotation=rotate_xticks)

        plt.show()
        plt.close()

def _check_ivars(data, ivars):

    '''
    Checks if the variables in the i() syntax are of the correct type.
    Args:
        data (pandas.DataFrame): The dataframe containing the data used for the model fitting.
        ivars (list): The list of variables specified in the i() syntax.
    Returns:
        None
    '''

    i0_type = data[ivars[0]].dtype
    i1_type = data[ivars[1]].dtype
    if not i0_type in ['category', "O"]:
        raise ValueError("Column " + ivars[0] + " is not of type 'O' or 'category', which is required in the first position of i(). Instead it is of type " +
                        i0_type.name + ". If a reference level is set, it is required that the variable in the first position of 'i()' is of type 'O' or 'category'.")
        if not i1_type in ['int64', 'float64', 'int32', 'float32']:
            raise ValueError("Column " + ivars[1] + " is not of type 'int' or 'float', which is required in the second position of i(). Instead it is of type " +
                            i1_type.name + ". If a reference level is set, iti is required that the variable in the second position of 'i()' is of type 'int' or 'float'.")


def _prepare_split_estimation(split, fsplit, data, fml_dict):

    '''
    Cleans the input for the split estimation.
    Checks if the split variables are of the correct type.

    Args:
        split (str): The name of the variable used for the split estimation.
        fsplit (str): The name of the variable used for the fixed split estimation.
        data (pandas.DataFrame): The dataframe containing the data used for the model fitting.
        var_dict (dict): The dictionary containing the variables used in the model.
    Returns:
        splitvar (pandas.Series): The series containing the split variable.
        splitvar_name (str): The name of the split variable. Either equal to split or fsplit.
        estimate_split_model (bool): Whether to estimate the split model.
        estimate_full_model (bool): Whether to estimate the full model.
    '''

    if split is not None:
        if fsplit is not None:
            raise ValueError(
                "Cannot specify both split and fsplit. Please specify only one of the two."
            )
        else:
            splitvar = data[split]
            estimate_full_model = False
            estimate_split_model = True
            splitvar_name = split
    elif fsplit is not None:
        splitvar = data[fsplit]
        splitvar_name = fsplit
        estimate_full_model = False
        estimate_split_model = True
    else:
        splitvar = None
        splitvar_name = None
        estimate_split_model = False
        estimate_full_model = True


    if splitvar is not None:
        split_categories = np.unique(splitvar)
        if splitvar_name not in data.columns:
            raise ValueError(
                "Split variable " +
                splitvar + " not found in data."
                )
        if splitvar_name in fml_dict.keys():
            raise ValueError(
                "Split variable " + splitvar +
                " cannot be a fixed effect variable."
            )
        if splitvar.dtype.name != "category":
            splitvar = pd.Categorical(splitvar)

    return splitvar, splitvar_name, estimate_split_model, estimate_full_model


def get_fml(y_names, x_names, fval):

    y_names = y_names[0]
    fml = y_names + " ~ " + "+".join(x_names)
    if fval != "0":
        fml += " | " + fval

    return fml.replace(" ", "")


def _multicollinearity_checks(X, Z, ivars, fml2):

    '''
    Checks for multicollinearity in the design matrices X and Z.
    Args:
        X (numpy.ndarray): The design matrix X.
        Z (numpy.ndarray): The design matrix (with instruments) Z.
        ivars (list): The list of variables specified in the i() syntax.
        fml2 (str): The formula string.

    '''

    if np.linalg.matrix_rank(X) < min(X.shape):
        if ivars is not None:
            raise MatrixNotFullRankError(
                'The design Matrix X does not have full rank for the regression with fml" + fml2 + "."'
                'The model is skipped.'
                'As you are running a regression via `i()` syntax, maybe you need to drop a level via i(var1, var2, ref = ...)?'
                )
        else:
            raise MatrixNotFullRankError(
                    'The design Matrix X does not have full rank for the regression with fml" + fml2 + "."'
                    'The model is skipped. '
                )

    if np.linalg.matrix_rank(Z) < min(Z.shape):
        if ivars is not None:
            raise MatrixNotFullRankError(
                'The design Matrix Z does not have full rank for the regression with fml" + fml2 + "."'
                'The model is skipped.'
                'As you are running a regression via `i()` syntax, maybe you need to drop a level via i(var1, var2, ref = ...)?"'
                )
        else:
            raise MatrixNotFullRankError(
                    'The design Matrix Z does not have full rank for the regression with fml" + fml2 + "."'
                    'The model is skipped.'
                )

def _get_vcov_type(vcov, fval):


    '''
    Passes the specified vcov type. If no vcov type specified, sets the default vcov type as iid if no fixed effect
    is included in the model, and CRV1 clustered by the first fixed effect if a fixed effect is included in the model.
    Args:
        vcov (str): The specified vcov type.
        fval (str): The specified fixed effects. (i.e. "X1+X2")
    Returns:
        vcov_type (str): The specified vcov type.
    '''

    if vcov is None:
        # iid if no fixed effects
        if fval == "0":
            vcov_type = "iid"
        else:
            # CRV1 inference, clustered by first fixed effect
            first_fe = fval.split("+")[0]
            vcov_type = {"CRV1": first_fe}
    else:
        vcov_type = vcov

    return vcov_type


def _clean_ivars(ivars, data):

    '''
    Clean variables interacted via i(X1, X2, ref = a) syntax.

    Args:
        ivars (list): The list of variables specified in the i() syntax.
        data (pandas.DataFrame): The dataframe containing the data used for the model fitting.
    Returns:
        ivars (list): The list of variables specified in the i() syntax minus the reference level
        drop_ref (str): The dropped reference level specified in the i() syntax. None if no level is dropped
    '''

    if ivars is not None:

        if list(ivars.keys())[0] is not None:
            ref = list(ivars.keys())[0]
            ivars = ivars[ref]
            drop_ref = ivars[0] + "[T." + ref + "]" + ":" + ivars[1]
        else:
            ivars = ivars[None]
            drop_ref = None

        # type checking for ivars variable
        _check_ivars(data, ivars)

    else:
        ivars = None
        drop_ref = None

    return ivars, drop_ref

def _drop_singletons(fixef_rm):

    '''
    Checks if the fixef_rm argument is set to "singleton". If so, returns True, else False.
    Args:
        fixef_rm (str): The fixef_rm argument.
    Returns:
        drop_singletons (bool): Whether to drop singletons.
    '''

    if fixef_rm == "singleton":
        return True
    else:
        return False



def _find_untransformed_depvar(transformed_depvar):

    '''
    Args:
        transformed_depvar (str): The transformed depvar

    find untransformed depvar in a formula
    i.e. if "a" is transormed to "log(a)", then "a" is returned
    '''

    match = re.search(r'\((.*?)\)', transformed_depvar)
    if match:
        return match.group(1)
    else:
        return transformed_depvar
