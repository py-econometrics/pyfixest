import warnings
import pyhdfe

import numpy as np
import pandas as pd

from typing import Any, Union, Dict, Optional
from scipy.stats import norm
from formulaic import model_matrix

from pyfixest.feols import Feols
from pyfixest.FormulaParser import FixestFormulaParser, _flatten_list


class Fixest:

    def __init__(self, data: pd.DataFrame) -> None:

        '''
        Initiate the fixest object.
        Deparse fml into formula dict, variable dict.

        Args:
            - data: The input pd.DataFrame for the object.

        Returns:
            - None
        '''

        self.data = data
        self.model_res = dict()


    def _demean(self):

        # deparse fxst.fml_dict:
        fixef_keys = list(self.var_dict.keys())

        self.demeaned_data_dict = dict()
        self.dropped_data_dict = dict()

        for f, fval in enumerate(fixef_keys):

            YX_dict = dict()
            na_dict = dict()

            if fval != "0":


                fval_list = fval.split("+")

                # find interacted fixed effects via "^"
                interacted_fes = [x for x in fval_list if len(x.split('^')) > 1]
                regular_fes = [x for x in fval_list if len(x.split('^')) == 1]

                for x in interacted_fes:
                    vars = x.split("^")
                    self.data[x] = self.data[vars].apply(lambda x: '^'.join(x.dropna().astype(str)) if x.notna().all() else np.nan, axis=1)

                for x in regular_fes:
                    self.data[x] = self.data[x].astype(str)

                fe = self.data[fval_list]
                # all fes to ints
                fe = fe.apply(lambda x: pd.factorize(x)[0])

                fe_na = np.sum(pd.isna(fe), axis = 1) > 0
                fe = np.array(fe)

                for fml in self.fml_dict[fval]:

                    Y, X = model_matrix(fml, self.data, na_action = 'ignore')
                    depvar = Y.columns
                    covars = X.columns

                    Y = np.array(Y)
                    X = np.array(X)

                    Y_na = np.isnan(Y).flatten()
                    X_na = np.sum(np.isnan(X), axis = 1) > 0

                    na_index = (Y_na + X_na) > 0
                    na_index = np.array(na_index + fe_na)
                    na_index = na_index.flatten()

                    Y = Y[~na_index]
                    X = X[~na_index]
                    fe2 = fe[~na_index]
                    # drop intercept
                    X = X[:,1:]

                    YX = np.concatenate([Y, X], axis = 1)

                    algorithm = pyhdfe.create(ids=fe2, residualize_method='map')
                    YX_demeaned = algorithm.residualize(YX)
                    YX_demeaned = pd.DataFrame(YX_demeaned)
                    YX_demeaned.columns = list(depvar) + list(covars[1:])

                    YX_dict[fml] = YX_demeaned
                    na_dict[fml] = na_index

            else:

                for fml in self.fml_dict[fval]:

                    Y, X = model_matrix(fml, self.data, na_action = 'ignore')
                    depvar = Y.columns
                    covars = X.columns

                    Y = np.array(Y)
                    X = np.array(X)

                    Y_na = np.isnan(Y).flatten()
                    X_na = np.sum(np.isnan(X), axis = 1) > 0

                    na_index = (Y_na + X_na) > 0

                    YX = np.concatenate([Y, X], axis = 1)
                    YX = YX[~na_index]
                    YX_demeaned = pd.DataFrame(YX)
                    YX_demeaned.columns = list(depvar) + list(covars)

                    YX_dict[fml] = YX_demeaned
                    na_dict[fml] = na_index

            self.demeaned_data_dict[fval] = YX_dict
            self.dropped_data_dict[fval] = na_dict

    def feols(self, fml: str, vcov: Union[str, Dict[str, str]]) -> None:
        '''
        Method for fixed effects regression modeling using PyHDFE package for projecting out fixed effects.

        Args:
            fml (str): A two-sided formula string using fixest formula syntax. Supported syntax includes:
                - Stepwise regressions (sw, sw0)
                - Cumulative stepwise regression (csw, csw0)
                - Multiple dependent variables (Y1 + Y2 ~ X)
                - Interacted fixed effects (e.g. "fe1^fe2" for interaction between fe1 and fe2)
                All other parts of the formula must be compatible with formula parsing via the formulaic module.
            vcov (str or dict): A string or dictionary specifying the type of variance-covariance matrix to use for inference.
                If a string, it can be one of "iid", "hetero", "HC1", "HC2", "HC3".
                If a dictionary, it should have the format {"CRV1":"clustervar"} for CRV1 inference or {"CRV3":"clustervar"} for CRV3 inference.

        Examples:
            - Standard formula:
                fml = 'Y ~ X1 + X2'
                fixest_model = Fixest(data=data).feols(fml, vcov='iid')
            - With fixed effects:
                fml = 'Y ~ X1 + X2 | fe1 + fe2'
            - With interacted fixed effects:
                fml = 'Y ~ X1 + X2 | fe1^fe2'
            - Multiple dependent variables:
                fml = 'Y1 + Y2 ~ X1 + X2'
            - Stepwise regressions (sw and sw0):
                fml = 'Y1 + Y2 ~ sw(X1, X2, X3)'
            - Cumulative stepwise regressions (csw and csw0):
                fml = 'Y1 + Y2 ~ csw(X1, X2, X3) '
            - Combinations:
                fml = 'Y1 + Y2 ~ csw(X1, X2, X3) | sw(X4, X5) + X6'
        '''

        fxst_fml = FixestFormulaParser(fml)

        fxst_fml.get_fml_dict()
        fxst_fml.get_var_dict()

        self.fml_dict = fxst_fml.fml_dict
        self.var_dict = fxst_fml.var_dict

        self._demean()

        for f, fval in enumerate(self.fml_dict.keys()):
            model_frames = self.demeaned_data_dict[fval]
            for x, fml in enumerate(model_frames):

                model_frame = model_frames[fml]
                Y = np.array(model_frame.iloc[:,0])
                X = model_frame.iloc[:,1:]
                colnames = X.columns
                X = np.array(X)
                FEOLS = Feols(Y, X)
                FEOLS.get_fit()
                FEOLS.na_index = self.dropped_data_dict[fval][fml]
                FEOLS.data = self.data[~FEOLS.na_index]
                FEOLS.get_vcov(vcov = vcov)
                FEOLS.get_inference()
                FEOLS.coefnames = colnames
                full_fml = fml + "|" + fval
                self.model_res[full_fml] = FEOLS

        return self


    def vcov(self, vcov: Union[str, Dict[str, str]]) -> None:

        '''
        Update inference on the fly. By calling vcov() on a "Fixest" object, all inference procedures applied
        to the "Fixest" object are replaced with the variance covariance matrix specified via the method.

        Args:
            - vcov: A string or dictionary specifying the type of variance-covariance matrix to use for inference.
                If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3".
                If a dictionary, it should have the format {"CRV1":"clustervar"} for CRV1 inference
                or {"CRV3":"clustervar"} for CRV3 inference.
        '''

        for model in list(self.model_res.keys()):

                fxst = self.model_res[model]

                fxst.get_vcov(vcov = vcov)
                fxst.get_inference()

        return self


    def tidy(self, type: Optional[str] = None) -> Union[pd.DataFrame, str]:

        '''
        Returns the results of an estimation using `feols()` as a tidy Pandas DataFrame.

        Parameters
        ----------
        type : str, optional
            The type of output format to use. If set to "markdown", the resulting DataFrame
            will be returned in a markdown format with three decimal places. Default is None.

        Returns
        -------
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
                        'coefnames':fxst.coefnames,
                        'Estimate': fxst.beta_hat,
                        'Std. Error': fxst.se,
                        't value': fxst.tstat,
                        'Pr(>|t|)': fxst.pvalue
                    }
                )
            )

        res = pd.concat(res, axis = 0).set_index('fml')
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
            - None

        '''

        for x in list(self.model_res.keys()):

            split = x.split("|")
            fe = split[1]
            depvar = split[0].split("~")[0]
            fxst = self.model_res[x]
            df = pd.DataFrame(
                  {
                      '':fxst.coefnames,
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
