from pyfixest.fixest import Fixest, Feols
import pandas as pd


def feols(fml, vcov, data):
    '''
    fixest function for regression modeling
    fixed effects are projected out via the PyHDFE package

    Args:

      fml (string, patsy Compatible): of the form Y ~ X1 + X2 | fe1 + fe2
      vcov (string or dict): either 'iid', 'hetero', 'HC1', 'HC2', 'HC3'. For
                             cluster robust inference, a dict {'CRV1':'clustervar'}
                             for CRV1 inference or {'CRV3':'clustervar'} for CRV3
                             inference.
      data (pd.DataFrame): DataFrame containing the data.

    Returns:

      A dictionariy with
        - estimated regression coefficients
        - standard errors
        - variance-covariance matrix
        - t-statistics
        - p-values
    '''

    fixest = Fixest(fml, data)
    fixest.demean()
    
    
    model_res = dict()
    for f, fval in enumerate(list(fixest.fml_dict.keys())):
        mf = fixest.demeaned_data_dict[fval]
        for fml in fixest.fml_dict[fval]:
            FEOLS = Feols(fml, mf)
            FEOLS.fit()
            FEOLS.vcov(vcov = vcov)
            FEOLS.inference()
            full_fml = fml + "|" + fval
            model_res[full_fml] = FEOLS
          

    res = []
    for x in list(model_res.keys()):
        
        fxst = model_res[x]
      
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
        
    res = pd.concat(res, axis = 0)

    return res
