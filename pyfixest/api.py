from pyfixest.fixest import fixest 
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

  
  
  fxst = fixest(fml, data)
  
  if fxst.has_fixef == True:
    fxst.do_demean() 
 
  fxst.do_fit()
  fxst.do_vcov(vcov = vcov)
  fxst.do_inference()
  
  res = []
  for x in range(0, fxst.n_regs):
    res.append(
      pd.DataFrame(
        {
        'depvar': fxst.depvars[x],
        'colnames' : fxst.coefnames, 
        'coef' : fxst.beta_hat[x], 
        'se' : fxst.se[x], 
        'tstat' : fxst.tstat[x], 
        'pvalue' : fxst.pvalue[x], 
        #'vcov' : fxst.vcov, 
        #'fixef_vars' : fxst.fixef_vars
        }
      )
    ) 
  
  return res

