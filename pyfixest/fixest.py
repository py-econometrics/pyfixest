import numpy as np
from scipy.stats import norm
import pyhdfe
from formulaic import model_matrix

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
        Y, X = model_matrix(fml_no_fixef, data)
        self.coefnames = X.columns
      else : 
        
        self.has_fixef = True
        self.fixef_vars = fml_split[1].replace(" ", "").split("+")
        fe = data[self.fixef_vars]
        self.fe = np.array(fe).reshape([self.N, len(self.fixef_vars)])
        Y, X = model_matrix(fml_no_fixef + '- 1', data)
        self.coefnames = X.columns

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
      self.tXy = (np.transpose(self.X) @ self.Y)
      beta_hat = self.tXXinv @ self.tXy
      self.beta_hat = beta_hat.flatten()
      self.Y_hat = (self.X @ self.beta_hat).reshape((self.N, 1))
      self.u_hat = self.Y - self.Y_hat

    def vcov(self, vcov):
      
      if isinstance(vcov, dict): 
        vcov_type_detail = list(vcov.keys())[0]
        self.clustervar = list(vcov.values())[0]
      elif isinstance(vcov, list):
        vcov_type_detail = vcov
      elif isinstance(vcov, str):
        vcov_type_detail = vcov
      else: 
        assert False, "arg vcov needs to be a dict, string or list"
        
      
      if vcov_type_detail == "iid": 
        vcov_type = "iid"
      elif vcov_type_detail in ["hetero", "HC1"]:
        vcov_type = "hetero"
      elif vcov_type_detail in ["CRV1", "CRV3"]:
        vcov_type = "CRV"
        
      # compute vcov
      if vcov_type == 'iid': 
        
        self.vcov = self.tXXinv * np.mean(self.u_hat ** 2) 
        
      elif vcov_type == 'hetero': 
        
        if vcov_type_detail == "HC1":
          self.ssc = self.N / (self.N  - self.k)
        else: 
          self.ssc = 1
        
        meat = np.transpose(self.X) * (self.u_hat.flatten() ** 2) @ self.X
        #meat = np.eye(self.N) * (self.u_hat ** 2)
        #meat = np.outer(score, score)
        self.vcov = self.ssc * self.tXXinv @ meat @  self.tXXinv
    
      elif vcov_type == "CRV":
        
        # if there are missings - delete them!
        cluster_df = self.data[self.clustervar]
        
        clustid = np.unique(cluster_df)
        self.G = len(clustid)
        
        if vcov_type_detail == "CRV1":
          
          meat = np.zeros((self.k, self.k))

          for igx, g, in enumerate(clustid):
          
            Xg = self.X[np.where(cluster_df == g)]
            ug = self.u_hat[np.where(cluster_df == g)]
            score_g = (np.transpose(Xg) @ ug).reshape((self.k, 1))
            meat += np.dot(score_g, score_g.transpose())
            
          self.ssc = self.G / (self.G - 1) * (self.N-1) / (self.N-self.k) 
          self.vcov = self.tXXinv @ meat @ self.tXXinv

        elif vcov_type_detail == "CRV3": 
          
          # check: is fixed effect cluster fixed effect? 
          # if not, either error or turn fixefs into dummies
          # for now: don't allow for use with fixed effects
          assert self.has_fixef == False, "CRV3 currently not supported with arbitrary fixed effects"
            
          beta_jack = np.zeros((self.G, self.k))
          tXX = np.transpose(self.X) @ self.X
          
          for ixg, g in enumerate(clustid):
            
            Xg = self.X[np.where(cluster_df == g)]
            Yg = self.Y[np.where(cluster_df == g)]
            tXgXg = np.transpose(Xg) @ Xg

            # jackknife regression coefficient
            beta_jack[ixg,:] = (
              np.linalg.pinv(tXX - tXgXg) @ (self.tXy - np.transpose(Xg) @ Yg)
            ).flatten()
            
          beta_center = self.beta_hat
          
          vcov = np.zeros((self.k, self.k))
          for ixg, g in enumerate(clustid):
            beta_centered = beta_jack[ixg,:] - beta_center
            vcov += np.outer(beta_centered, beta_centered)
          
          self.ssc = self.G / (self.G - 1)
          self.vcov = self.ssc * vcov

    def inference(self):
  
      self.se = np.sqrt(np.diagonal(self.vcov))
      self.tstat = self.beta_hat / self.se
      self.pvalue = 2*(1-norm.cdf(np.abs(self.tstat)))
      
    def performance(self):
      
      self.r_squared = 1 - np.sum(self.u_hat ** 2) / np.sum((self.Y - np.mean(self.Y))**2)
      self.adj_r_squared = (self.N - 1) / (self.N - self.k) * self.r_squared
      

    
  
    
  
