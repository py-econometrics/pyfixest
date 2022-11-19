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

      self.Y = np.array(Y)
      self.X = np.array(X)
      self.k = X.shape[1]


  
    def demean(self): 
      
      algorithm = pyhdfe.create(ids = self.fe, residualize_method = 'map')
      YX = np.concatenate([self.Y,self.X], axis = 1)
      residualized = algorithm.residualize(YX)
      self.Y = residualized[:, [0]]
      self.X = residualized[:, 1:]
      self.k = self.X.shape[1]

    def fit(self):
      
      # k without fixed effects
      #N, k = X.shape
      
      self.tXXinv = np.linalg.inv(np.transpose(self.X) @ self.X)
      beta_hat = self.tXXinv @ (np.transpose(self.X) @ self.Y)
      self.beta_hat = beta_hat.flatten()
      self.Y_hat = (self.X @ self.beta_hat).reshape((self.N, 1))
      self.u_hat = self.Y - self.Y_hat

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
        
        meat = np.transpose(self.X) * (self.u_hat.flatten() ** 2) @ self.X
        #meat = np.eye(self.N) * (self.u_hat ** 2)
        #meat = np.outer(score, score)
        self.vcov = self.ssc * self.tXXinv @ meat @  self.tXXinv
    
      else:
        
        cluster = vcov
        cluster_df = data[cluster]
        
        clustid = np.unique(cluster_df)
        self.G = len(clustid)
        
        meat = np.zeros((self.k, self.k))
        for igx, g, in enumerate(clustid):
          Xg = self.X[np.where(cluster_df == g)]
          ug = self.u_hat[np.where(cluster_df == g)]
          score_g = (np.transpose(self.X) @ self.u_hat).reshape((self.k, 1))
          meat += np.dot(score_g, score_g.transpose())
          
        self.ssc = self.G / (self.G - 1) * (self.N-1) / (self.N-self.k) 
        self.vcov = self.tXXinv @ meat @ self.tXXinv
        
        
    def inference(self):
  
      self.se = np.sqrt(np.diagonal(self.vcov))
      self.tstat = self.beta_hat / self.se
      self.pvalue = 2*(1-norm.cdf(np.abs(self.tstat)))
      
    def performance(self):
      
      self.r_squared = 1 - np.sum(self.u_hat ** 2) / np.sum((self.Y - np.mean(self.Y))**2)
      self.adj_r_squared = (self.N - 1) / (self.N - self.k) * self.r_squared
      

    
  
