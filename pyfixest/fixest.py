import warnings
import pyhdfe

import numpy as np
import pandas as pd
from scipy.stats import norm
from formulaic import model_matrix
from tabulate import tabulate


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


            cols = self.var_dict[fval]
            YX_df = self.data[cols]
            YX_na_index = pd.isna(YX_df)
            n = YX_df.shape[0]

            # deparse fml dict
            var_dict2 = dict()
            for x in self.fml_dict[fval]:
              variables = x.split('~')[1].split('+')
              variables.insert(0, x.split('~')[0])
              var_dict2[x] = variables

            n_fml = len(self.fml_dict[fval])
            fml_na_index = np.zeros((n, n_fml))

            var_list = list(var_dict2.values())

            for x, xval in enumerate(var_list):
                fml_na_index[:,x] = YX_na_index[xval].sum(axis = 1).values

            fml_na_index = fml_na_index.astype(bool)

            fml_na_max = np.argsort(fml_na_index.sum(axis = 0))
            # here: optimize by going from 'most missings to fewest'

            # drop NAs
            YX_dict = dict()
            for x, xval in enumerate(var_list):

              YX = YX_df[xval].dropna()

              if fval != "0":
                  fval_list = fval.split("+")
                  fe = np.array(self.data[fval_list], dtype = 'float64')
                  if fe.ndim == 1:
                      fe.shape = (len(fe), 1)

                  # drop missing YX from fe
                  fe = fe[~fml_na_index[:,x]]
                  # drop data with missing fe's
                  fe_na = np.mean(np.isnan(fe), axis = 1) > 0
                  fe = fe[~fe_na]
                  fe = fe.astype(int)

                  YX = YX.iloc[~fe_na, :]
                  na_deleted = sum(fe_na)
                  self.dropped_data_dict[fval] = na_deleted
                  
                  fml = list(var_dict2.keys())[x]
                  Y, X = model_matrix(fml + "-1", YX)
                  YX = pd.concat([Y,X], axis = 1)
                  colnames = YX.columns
                  YX = np.array(YX)
  
                  algorithm = pyhdfe.create(ids=fe, residualize_method='map')
                  data_demean = algorithm.residualize(YX)
                  data_demean = pd.DataFrame(data_demean)
                  data_demean.columns = colnames

                  
                  # return as pd.DataFrame

              else:
                  # no need to drop missing fe + no need to drop intercept
                  fml = list(var_dict2.keys())[x]
                  Y, X = model_matrix(fml, YX)
                  data_demean = pd.concat([Y,X], axis = 1)
                  
              data_demean = pd.DataFrame(data_demean)
              YX_dict[list(var_dict2.keys())[x]] = data_demean

            self.demeaned_data_dict[fval] = YX_dict
 
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
      
        fml = FixestFormulaParser(fml)
        
        fml.get_fml_dict()
        fml.get_var_dict()
        
        self.fml_dict = fml.fml_dict
        self.var_dict = fml.var_dict
        
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
                FEOLS.fit()
                FEOLS.data = self.data
                FEOLS.vcov(vcov = vcov)
                FEOLS.inference()
                FEOLS.coefnames = colnames
                full_fml = fml + "|" + fval
                self.model_res[full_fml] = FEOLS
            
    def summary(self, type = None): 
      
        res = []
        for x in list(self.model_res.keys()):
            
            fxst = self.model_res[x]
          
            res.append(
                pd.DataFrame(
                    {
                        'fml': x,
                        'coefnames':fxst.coefnames, 
                        'coef': fxst.beta_hat,
                        'se': fxst.se,
                        'tstat': fxst.tstat,
                        'pvalue': fxst.pvalue
                    }
                )
            )
        
        res = pd.concat(res, axis = 0).set_index('fml')
        if type == "markdown": 
            return res.to_markdown(floatfmt=".3f")
        else: 
            return res
