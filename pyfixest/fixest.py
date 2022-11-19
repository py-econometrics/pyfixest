import numpy as np
import patsy
from scipy.stats import norm
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
        self.coefnames = X.design_info.column_names
      else : 
        
        self.has_fixef = True
        self.fixef_vars = fml_split[1].replace(" ", "").split("+")
        fe = data[self.fixef_vars]
        self.fe = np.array(fe).reshape([self.N, len(self.fixef_vars)])
        Y, X = patsy.dmatrices(fml_no_fixef + '- 1', data)
        self.coefnames = X.design_info.column_names

      self.Y = np.array(Y).flatten()
      self.X = np.array(X)
      self.k = X.shape[1]


  
    def demean(self): 
      
      algorithm = pyhdfe.create(ids = self.fe, residualize_method = 'map')
      YX = np.concatenate([self.Y,self.X], axis = 1)
      residualized = algorithm.residualize(YX)
      self.Y = residualized[:, [0]].flatten()
      self.X = residualized[:, 1:]
      self.k = self.X.shape[1]

    def fit(self):
      
      # k without fixed effects
      #N, k = X.shape
      
      self.tXXinv = np.linalg.inv(self.X.transpose() @ self.X)
      beta_hat = self.tXXinv @ (self.X.transpose() @ self.Y)
      self.beta_hat = beta_hat.flatten()
      Y_predict = (self.X @ self.beta_hat).flatten()
      self.u_hat = self.Y - Y_predict

    def vcov(self, vcov):
      
      # compute vcov
      if vcov == 'iid': 
        
        self.vcov = self.tXXinv * np.mean(self.u_hat ** 2) 
        
      elif vcov == 'hetero': 
        
        vcov_type = "HC1"
        if vcov_type == "HC1":
          self.ssc = self.N / (self.N  - self.k)
        else: 
          self.ssc = 1
        
        score = (self.X.transpose() @ self.u_hat)
        meat = score @ score.transpose()
        self.vcov = self.ssc * self.tXXinv * np.diag(meat) @  self.tXXinv
    
      else:
        
        cluster = vcov
        cluster_df = data[cluster]
        
        clustid = np.unique(cluster_df)
        self.G = len(clustid)
        
        meat = np.zeros((self.k, self.k))
        for igx, g, in enumerate(clustid):
          Xg = self.X[np.where(cluster_df == g)]
          ug = self.u_hat[np.where(cluster_df == g)]
          score_g = (self.X.transpose() @ self.u_hat).reshape((self.k, 1))
          meat += np.dot(score_g, score_g.transpose())
          
        self.ssc = self.G / (self.G - 1) * (self.N-1) / (self.N-self.k) 
        self.vcov = self.tXXinv @ meat @ self.tXXinv
        
        
    def inference(self):
  
      self.se = np.sqrt(np.diagonal(self.vcov))
      self.tstat = self.beta_hat / self.se
      self.pvalue = 2*(1-norm.cdf(np.abs(self.tstat)))


    
  
