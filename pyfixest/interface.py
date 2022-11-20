from pyfixest import fixest 

def feols(fml, vcov, data):
  
  '''
  fixest function for regression modeling
  fixed effects are projected out via the PyHDFE package
  
  Args: 
    
    fml (string, patsy Compatible): of the form Y ~ X1 + X2 | fe1 + fe2
    vcov (string or named dictionary): either 'iid' or 'hetero'. 
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
    fxst.demean() 
 
  fxst.fit()
  fxst.vcov(vcov = vcov)
  fxst.inference()
  
  res = {
    'coef' : fxst.beta_hat, 
    'se' : fxst.se, 
    'tstat' : fxst.tstat, 
    'pvalue' : fxst.pvalue, 
    'vcov' : fxst.vcov, 
    'fixef_vars' : fxst.fixef_vars
    }
    
  return res

