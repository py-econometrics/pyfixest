import warnings
import pyhdfe

import numpy as np
import pandas as pd
from scipy.stats import norm
from formulaic import model_matrix

from pyfixest.feols import Feols
from pyfixest.FormulaParser import FixestFormulaParser, _flatten_list


class Fixest:

    def __init__(self, data):

        '''
        Initiate the fixest object.
        Deparse fml into formula dict, variable dict.
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

    def feols(self, fml, vcov):

        '''
        fixest function for regression modeling
        fixed effects are projected out via the PyHDFE package

        Args:

          fml (string, patsy Compatible): of the form Y ~ X1 + X2 | fe1 + fe2
          vcov (string or dict): either 'iid', 'hetero', 'HC1', 'HC2', 'HC3'. For
                                 cluster robust inference, a dict {'CRV1':'clustervar'}
                                 for CRV1 inference or {'CRV3':'clustervar'} for CRV3
                                 inference.

        Returns:

          A dictionariy with
            - estimated regression coefficients
            - standard errors
            - variance-covariance matrix
            - t-statistics
            - p-values
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


    def vcov(self, vcov):

      '''
      update inference on the fly
      '''

      for model in list(self.model_res.keys()):

            fxst = self.model_res[model]

            fxst.get_vcov(vcov = vcov)
            fxst.get_inference()

      return self


    def tidy(self, type = None):

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

    def summary(self):


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
