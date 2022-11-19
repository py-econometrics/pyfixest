import numpy as np
import patsy
from scipy.stats import t
import pyhdfe

class fixest:
  
    def __init__(self, fml, data):
  
      self.data = data
      self.N = data.shape[0]
      fml_split = fml.split("|")
      fml_no_fixef = fml_split[0].strip()
    
      if len(fml_split) == 1: 
        # if length = 1, then no fixed effect
        self.has_fixef = False
        self.fixef_vars = None
        Y, X = patsy.dmatrices(fml_no_fixef, data)

      else : 
        
        self.has_fixef = True
        self.fixef_vars = fml_split[1].replace(" ", "").split("+")
        fe = data[self.fixef_vars]
        self.fe = np.array(fe).reshape([self.N, len(self.fixef_vars)])
        Y, X = patsy.dmatrices(fml_no_fixef + '- 1', data)

      self.Y = np.array(Y)
      self.X = np.array(X)
      self.k = X.shape[1]


  
    def demean(self): 
      
      algorithm = pyhdfe.create(ids = self.fe, residualize_method = 'map')
      YX = np.concatenate([self.Y,self.X], axis = 1)
      residualized = algorithm.residualize(YX)
      self.Y = residualized[:, [0]]
      self.X = residualized[:, 1:]

    def fit(self):
      
      # k without fixed effects
      #N, k = X.shape
      
      self.XXinv = np.linalg.inv(self.X.transpose() @ self.X)
      beta_hat = self.XXinv @ (self.X.transpose() @ self.Y)
      self.beta_hat = beta_hat.flatten()
      Y_predict = self.X @ self.beta_hat
      self.u_hat = self.Y - Y_predict      

    def vcov(self, vcov):
      
      # compute vcov
      if vcov == 'iid': 
        self.vcov = self.XXinv * np.mean(self.u_hat ** 2) 
      elif vcov == 'hetero': 
        score = self.X.transpose() @ self.u_hat
        self.vcov = self.XXinv @ (score @ score.transpose()) @  self.XXinv.transpose()
    
    def inference(self, dof = 'default'):
  
      if dof == 'default':
        dof = self.N - self.k
  
      self.se = np.sqrt(np.diagonal(self.vcov))
      self.tstat = self.beta_hat / self.se
      self.pvalue = 2*(1-t.cdf(self.tstat, df = dof))


    
  
