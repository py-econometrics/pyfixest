import pyhdfe
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

from typing import Any, Union, Dict, Optional, List, Tuple
from scipy.stats import norm
from formulaic import model_matrix

from plotnine import ggplot, aes, geom_errorbar, geom_point, theme_bw, ylab, xlab, geom_hline, position_dodge

from pyfixest.feols import Feols
from pyfixest.FormulaParser import FixestFormulaParser, _flatten_list


class Fixest:

    def __init__(self, data: pd.DataFrame) -> None:
        '''

        A class for fixed effects regression modeling.

        Args:
            data: The input pd.DataFrame for the object.

        Returns:
            None
        '''

        self.data = data
        self.model_res = dict()

    def _clean_fe(self, fval):

        fval_list = fval.split("+")

        # find interacted fixed effects via "^"
        interacted_fes = [
            x for x in fval_list if len(x.split('^')) > 1]

        for x in interacted_fes:
            vars = x.split("^")
            self.data[x] = self.data[vars].apply(lambda x: '^'.join(
                x.dropna().astype(str)) if x.notna().all() else np.nan, axis=1)

        fe = self.data[fval_list]
        # all fes to factors / categories
        fe = fe.apply(lambda x: pd.factorize(x)[0])

        fe_na = fe.isna().any(axis=1)
        fe = fe.to_numpy()

        return fe, fe_na


    def demean(self, fval, ivars, drop_ref):

        '''
        demean all regressions for a specification of fixed effects

        Args:
            fval: A specification of fixed effects. String. E.g. X4 or
                "X3 + X2"
            ivars: the interacted variables
            drop_ref: the reference category for the interacted variables.
        '''

        YX_dict = dict()
        na_dict = dict()

        if fval != "0":
            fe, fe_na = self._clean_fe(fval)
            fe_na = list(fe_na[fe_na == True])
        else:
            fe = None
            fe_na = None

        dict2fe = self.fml_dict2.get(fval)

        # create lookup table with NA index key
        # for each regression, check if lookup table already
        # populated with "demeaned" data for some (or all)
        # variables of the model. if demeaned variable for
        # NA key already exists -> use it. else rerun demeaning
        # algorithm

        lookup_demeaned_data = dict()


        for depvar in dict2fe.keys():

            # [(0, 'X1+X2'), (1, ['X1+X3'])]
            for c, covar in enumerate(dict2fe.get(depvar)):

                if isinstance(covar, list):
                    covar2 = covar[0]
                else:
                    covar2 = covar

                if isinstance(depvar, list):
                    depvar2 = depvar[0]
                else:
                    depvar2 = depvar

                fml = depvar2 + " ~ " + covar2

                Y, X = model_matrix(fml, self.data)
                na_index = list(set(range(0, self.data.shape[0])) - set(Y.index))

                if self.ivars is not None:
                    if drop_ref is not None:
                        X = X.drop(drop_ref, axis=1)

                dep_varnames = list(Y.columns)
                co_varnames = list(X.columns)
                var_names = list(dep_varnames) + list(co_varnames)

                if self.ivars is not None:
                    self.icovars = [s for s in co_varnames if s.startswith(
                        ivars[0]) and s.endswith(ivars[1])]
                else:
                    self.icovars = None

                Y = Y.to_numpy()
                X = X.to_numpy()

                # variant 1: if there are fixed effects to be projected out
                if fe is not None:
                    na_index = (na_index + fe_na)
                    fe2 = np.delete(fe, na_index, axis = 0)
                    # drop intercept
                    X = X[:, 1:]
                    co_varnames.remove("Intercept")
                    var_names.remove("Intercept")

                    # check if variables have already been demeaned
                    Y = np.delete(Y, fe_na, axis = 0)
                    X = np.delete(X, fe_na, axis = 0)
                    YX = np.concatenate([Y,X], axis = 1)

                    na_index_str = ','.join(str(x) for x in na_index)

                    # check if looked dict has data for na_index
                    if lookup_demeaned_data.get(na_index_str) is not None:
                        # get data out of lookup table: list of [algo, data]
                        algorithm, YX_demeaned_old = lookup_demeaned_data.get(na_index_str)

                        # get not yet demeaned covariates
                        var_diff_names = list(set(var_names) - set(YX_demeaned_old.columns))[0]
                        var_diff_index = list(var_names).index(var_diff_names)
                        var_diff = YX[:,var_diff_index]
                        if var_diff.ndim == 1:
                            var_diff = var_diff.reshape(len(var_diff), 1)


                        YX_demean_new = algorithm.residualize(var_diff)
                        YX_demeaned = np.concatenate([YX_demeaned_old, YX_demean_new], axis = 1)
                        YX_demeaned = pd.DataFrame(YX_demeaned)

                        YX_demeaned.columns = list(YX_demeaned_old.columns) + [var_diff_names]


                    else:
                        # not data demeaned yet for NA combination
                        algorithm = pyhdfe.create(ids=fe2, residualize_method='map')
                        YX_demeaned = algorithm.residualize(YX)
                        YX_demeaned = pd.DataFrame(YX_demeaned)
                        YX_demeaned.columns = list(dep_varnames) + list(co_varnames)

                    lookup_demeaned_data[na_index_str] = [algorithm, YX_demeaned]

                else:
                    # if no fixed effects
                    YX = np.concatenate([Y, X], axis=1)
                    YX_demeaned = pd.DataFrame(YX)
                    YX_demeaned.columns = list(dep_varnames) + list(co_varnames)

                YX_dict[fml] = YX_demeaned
                na_dict[fml] = na_index

        return YX_dict, na_dict


    def feols(self, fml: str, vcov: Union[str, Dict[str, str]]) -> None:
        '''
        Method for fixed effects regression modeling using the PyHDFE package for projecting out fixed effects.

        Args:
            fml (str): A two-sided formula string using fixest formula syntax. Supported syntax includes:
                Stepwise regressions (sw, sw0)
                Cumulative stepwise regression (csw, csw0)
                Multiple dependent variables (Y1 + Y2 ~ X)
                Interacted fixed effects (e.g. "fe1^fe2" for interaction between fe1 and fe2)
                All other parts of the formula must be compatible with formula parsing via the formulaic module.
            vcov (Union(str, dict)): A string or dictionary specifying the type of variance-covariance matrix to use for inference.
                If a string, it can be one of "iid", "hetero", "HC1", "HC2", "HC3".
                If a dictionary, it should have the format dict("CRV1":"clustervar") for CRV1 inference or dict(CRV3":"clustervar") for CRV3 inference.

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
        '''

        fxst_fml = FixestFormulaParser(fml)

        fxst_fml.get_fml_dict()
        fxst_fml.get_var_dict()
        fxst_fml._transform_fml_dict()

        self.fml_dict = fxst_fml.fml_dict
        self.var_dict = fxst_fml.var_dict
        self.fml_dict2 = fxst_fml.fml_dict2
        self.ivars = fxst_fml.ivars

        # get all fixed effects combinations
        fixef_keys = list(self.var_dict.keys())

        if self.ivars is not None:

            if list(self.ivars.keys())[0] is not None:
                ref = list(self.ivars.keys())[0]
                ivars = self.ivars[ref]
                drop_ref = ivars[0] + "[T." + ref + "]" + ":" + ivars[1]
            else:
                ivars = self.ivars[None]
                drop_ref = None

            # type checking
            i0_type = self.data[ivars[0]].dtype
            i1_type = self.data[ivars[1]].dtype
            if not i0_type in ['category', "O"]:
                raise ValueError("Column " + ivars[0] + " is not of type 'O' or 'category', which is required in the first position of i(). Instead it is of type " + i0_type.name + ". If a reference level is set, it is required that the variable in the first position of 'i()' is of type 'O' or 'category'.")
            if not i1_type in ['int64', 'float64', 'int32', 'float32']:
                raise ValueError("Column " + ivars[1] + " is not of type 'int' or 'float', which is required in the second position of i(). Instead it is of type " + i1_type.name + ". If a reference level is set, iti is required that the variable in the second position of 'i()' is of type 'int' or 'float'.")

        else:
            ivars = None
            drop_ref = None



        self.dropped_data_dict = dict()
        self.demeaned_data_dict = dict()

        for _, fval in enumerate(fixef_keys):
            self.demeaned_data_dict[fval], self.dropped_data_dict[fval] = self.demean(fval, ivars, drop_ref)



        # estimate models based on demeaned model matrix and dependent variables
        for _, fval in enumerate(self.fml_dict.keys()):
            model_frames = self.demeaned_data_dict[fval]
            for _, fml in enumerate(model_frames):

                model_frame = model_frames[fml]
                full_fml = fml + "|" + fval

                Y = model_frame.iloc[:, 0].to_numpy()
                X = model_frame.iloc[:, 1:]
                colnames = X.columns
                X = X.to_numpy()

                if np.linalg.matrix_rank(X) < min(X.shape):
                    raise ValueError("The design Matrix X does not have full rank for the regression with fml" + full_fml + ". The model is skipped. If you are running a regression via `i()` syntax, maybe you need to drop a level via i(var1, var2, ref = ...)?")

                FEOLS = Feols(Y, X)
                FEOLS.get_fit()
                FEOLS.na_index = self.dropped_data_dict[fval][fml]
                FEOLS.data = self.data.iloc[~self.data.index.isin(FEOLS.na_index), :]
                FEOLS.get_nobs()
                FEOLS.get_vcov(vcov=vcov)
                FEOLS.get_inference()
                FEOLS.coefnames = colnames
                if self.icovars is not None:
                    FEOLS.icovars = self.icovars
                self.model_res[full_fml] = FEOLS

        return self

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

        for model in list(self.model_res.keys()):

            fxst = self.model_res[model]

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
        for x in list(self.model_res.keys()):

            fxst = self.model_res[x]

            res.append(
                pd.DataFrame(
                    {
                        'fml': x,
                        'coefnames': fxst.coefnames,
                        'Estimate': fxst.beta_hat,
                        'Std. Error': fxst.se,
                        't value': fxst.tstat,
                        'Pr(>|t|)': fxst.pvalue
                    }
                )
            )

        res = pd.concat(res, axis=0).set_index('fml')
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

        for x in list(self.model_res.keys()):

            split = x.split("|")
            fe = split[1]
            depvar = split[0].split("~")[0]
            fxst = self.model_res[x]
            df = pd.DataFrame(
                {
                    '': fxst.coefnames,
                    'Estimate': fxst.beta_hat,
                    'Std. Error': fxst.se,
                    't value': fxst.tstat,
                    'Pr(>|t|)': fxst.pvalue
                }
            )

            print('')
            print('### Fixed-effects:', fe)
            print('Dep. var.:', depvar)
            print('')
            print(df.to_string(index=False))
            print('---')

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

        icovars = self.icovars


        if icovars is None:
            raise ValueError(
                "The estimated models did not have ivars / 'i()' model syntax. In consequence, the '.iplot()' method is not supported.")

        # ivars_keys = self.ivars.keys()
        # if ivars_keys is not None:
        #     ref = list(ivars_keys)[0]
        # else:
        #     ref = None
        #
        # if "Intercept" in icovars:
        #     icovars.remove("Intercept")

        df = self.tidy()

        df = df[df.coefnames.isin(icovars)]
        models = df.index.unique()

        df_list = []

        for model in models:

            df_model = df.xs(model)
            coef = df_model["Estimate"].values
            conf_l = coef - df_model["Std. Error"].values * norm.ppf(1 - alpha / 2)
            conf_u = coef + df_model["Std. Error"].values  * norm.ppf(1 - alpha / 2)
            coefnames = df_model["coefnames"].values.tolist()

            coefnames = [(i) for string in coefnames for i in re.findall(
                r'\[T\.([\d\.\-]+)\]', string)]

            if ref is not None:
                coef = np.append(coef, 0)
                conf_l = np.append(conf_l, 0)
                conf_u = np.append(conf_u, 0)
                coefnames = np.append(coefnames, ref)

            df_dict = {
                'coef': coef,
                'conf_l': conf_l,
                'conf_u': conf_u,
                'coefnames': coefnames,
                'model': model
            }

            df_list.append(pd.DataFrame(df_dict))

        df_all = pd.concat(df_list, axis=0)

        iplot = (
            ggplot(df_all, aes(x='coefnames', y='coef', color='model', group = 'model')) +
            geom_point(position = position_dodge(0.5)) +
            geom_errorbar(aes(x='coefnames', ymin='conf_l', ymax='conf_u'), position = position_dodge(0.5)) +
            theme_bw() +
            ylab('Estimate') +
            xlab(list(self.ivars.values())[0][0]) +
            geom_hline(yintercept=0, color="blue", linetype="dotted")
        )


        return iplot

    def coefplot(self, alpha = 0.05, figsize=(5, 2), yintercept=0, figtitle=None, figtext=None):
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

        df = self.tidy()
        models = df.index.unique()

        _coefplot(
            models = models,
            figsize = figsize,
            alpha = alpha,
            yintercept = yintercept,
            xintercept = None,
            df = df,
            is_iplot = False,
            rotate_xticks=rotate_xticks
        )




def _coefplot(models: List, df: pd.DataFrame, figsize: Tuple[int, int], alpha: float, yintercept: Optional[int] = None,
              xintercept: Optional[int] = None, is_iplot: bool = False,
              rotate_xticks: float = 0) -> None:
    """
        Plot model coefficients with confidence intervals.

        Args:
            models (list): A list of fitted models.
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

        fig, ax = plt.subplots(len(models), gridspec_kw={'hspace': 0.5}, figsize=figsize)
        fig.suptitle("iplot")

        for x, model in enumerate(models):

            df_model = df.xs(model)
            coef = df_model["Estimate"].values
            conf_l = coef - df_model["Std. Error"].values * norm.ppf(1 - alpha / 2)
            conf_u = coef + df_model["Std. Error"].values  * norm.ppf(1 - alpha / 2)
            coefnames = df_model["coefnames"].values.tolist()

            # could be moved out of the for loop, as the same ivars for all
            # models.

            if is_iplot == True:
                coefnames = [(i) for string in coefnames for i in re.findall(
                    r'\[T\.([\d\.\-]+)\]', string)]

            # in the future: add reference category
            #if ref is not None:
            #    coef = np.append(coef, 0)
            #    conf_l = np.append(conf_l, 0)
            #    conf_u = np.append(conf_u, 0)
            #    coefnames = np.append(coefnames, ref)

            ax[x].scatter(coefnames,coef, color= "b", alpha = 0.8)
            ax[x].scatter(coefnames,conf_u, color = "b", alpha = 0.8, marker = "_", s = 100)
            ax[x].scatter(coefnames,conf_l, color = "b", alpha = 0.8, marker = "_", s = 100)
            ax[x].vlines(coefnames, ymin=conf_l, ymax=conf_u, color= "b", alpha = 0.8)
            if yintercept is not None:
                ax[x].axhline(yintercept, color='red', linestyle='--', alpha = 0.5)
            if xintercept is not None:
                ax[x].axvline(xintercept, color='red', linestyle='--', alpha = 0.5)
            ax[x].set_ylabel('Coefficients')
            ax[x].set_title(model)
            ax[x].tick_params(axis='x', rotation=rotate_xticks)

    else:

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle("iplot")

        model = models[0]

        df_model = df.xs(model)
        coef = df_model["Estimate"].values
        conf_l = coef - df_model["Std. Error"].values * norm.ppf(1 - alpha / 2)
        conf_u = coef + df_model["Std. Error"].values  * norm.ppf(1 - alpha / 2)
        coefnames = df_model["coefnames"].values.tolist()

        if is_iplot == True:
            coefnames = [(i) for string in coefnames for i in re.findall(
                r'\[T\.([\d\.\-]+)\]', string)]

        # in the future: add reference category
        #if ref is not None:
        #    coef = np.append(coef, 0)
        #    conf_l = np.append(conf_l, 0)
        #    conf_u = np.append(conf_u, 0)
        #    coefnames = np.append(coefnames, ref)

                #c = next(color)

        ax.scatter(coefnames,coef, color= "b", alpha = 0.8)
        ax.scatter(coefnames,conf_u, color = "b", alpha = 0.8, marker = "_", s = 100)
        ax.scatter(coefnames,conf_l, color = "b", alpha = 0.8, marker = "_", s = 100)
        ax.vlines(coefnames, ymin=conf_l, ymax=conf_u, color= "b", alpha = 0.8)
        if yintercept is not None:
            ax.axhline(yintercept, color='red', linestyle='--', alpha = 0.5)
        if xintercept is not None:
            ax.axvline(xintercept, color='red', linestyle='--', alpha = 0.5)
        ax.set_ylabel('Coefficients')
        ax.set_title(model)
        ax.tick_params(axis='x', rotation=rotate_xticks)


        plt.show()
        plt.close()
