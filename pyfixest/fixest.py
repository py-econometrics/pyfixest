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
        #Y, X = patsy.dmatrices(fml_no_fixef, data)
        #self.coefnames = X.design_info.column_names

      else : 
        
        self.has_fixef = True
        self.fixef_vars = fml_split[1].replace(" ", "").split("+")
        fe = data[self.fixef_vars]
        self.fe = np.array(fe).reshape([self.N, len(self.fixef_vars)])
        Y, X = model_matrix(fml_no_fixef + '- 1', data)
        self.coefnames = X.columns
        #Y, X = patsy.dmatrices(fml_no_fixef + '- 1', data)
        #self.coefnames = X.design_info.column_names
      
      self.depvars = Y.columns  
      self.Y = np.array(Y)
      self.X = np.array(X)
      self.k = X.shape[1]
      self.n_regs = self.Y.shape[1]

  
    def demean(self): 
      
      algorithm = pyhdfe.create(ids = self.fe, residualize_method = 'map')
      YX = np.concatenate([self.Y,self.X], axis = 1)
      residualized = algorithm.residualize(YX)
      self.Y = residualized[:, :self.n_regs]
      self.X = residualized[:, self.n_regs:]
      self.k = self.X.shape[1]

    def fit(self):
      
      # k without fixed effects
      #N, k = X.shape
      self.tXXinv = np.linalg.inv(np.transpose(self.X) @ self.X)
      
      self.tXy = []
      self.beta_hat = []
      self.Y_hat = []
      self.u_hat = []
      
      # loop over all dependent variables
      for regs in range(0, self.n_regs):
        self.tXy.append(np.transpose(self.X) @ self.Y[:,regs])
        beta_hat = self.tXXinv @ self.tXy[regs]
        self.beta_hat.append(beta_hat.flatten())
        self.Y_hat.append((self.X @ self.beta_hat[regs]).reshape((self.N, 1)))
        self.u_hat.append(self.Y[:,regs] - self.Y_hat[regs].flatten())

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
      elif vcov_type_detail in ["hetero", "HC1", "HC2", "HC3"]:
        vcov_type = "hetero"
      elif vcov_type_detail in ["CRV1", "CRV3"]:
        vcov_type = "CRV"
       
      self.vcov = []
      self.ssc = []
      for x in range(0,self.n_regs) :   
        # compute vcov
        if vcov_type == 'iid': 
          
          self.vcov.append(self.tXXinv * np.mean(self.u_hat[x] ** 2))
          
        elif vcov_type == 'hetero': 
          
          if vcov_type_detail in ["hetero", "HC1"]:
            self.ssc.append(self.N / (self.N  - self.k))
            u = self.u_hat[x].flatten()
          elif vcov_type_detail in ["HC2", "HC3"]:
            self.ssc.append(1)
            leverage = np.mean(self.X * (self.X @ self.tXXinv), axis = 1)
            if vcov_type_detail == "HC2":
              u = (1 - leverage) * self.u_hat[x].flatten()
            else:
              u = np.sqrt(1 - leverage) * self.u_hat[x].flatten()
  
          meat = np.transpose(self.X) * (u ** 2) @ self.X
          self.vcov.append(
            self.ssc[x] * self.tXXinv @ meat @  self.tXXinv
          ) 
      
        elif vcov_type == "CRV":
          
          # if there are missings - delete them!
          cluster_df = self.data[self.clustervar]
          
          clustid = np.unique(cluster_df)
          self.G = len(clustid)
          
          if vcov_type_detail == "CRV1":
            
            meat = np.zeros((self.k, self.k))
  
            for igx, g, in enumerate(clustid):
            
              Xg = self.X[np.where(cluster_df == g)]
              ug = self.u_hat[x][np.where(cluster_df == g)]
              score_g = (np.transpose(Xg) @ ug).reshape((self.k, 1))
              meat += np.dot(score_g, score_g.transpose())
              
            self.ssc.append(
              self.G / (self.G - 1) * (self.N-1) / (self.N-self.k)
            )
            self.vcov.append(
              self.ssc * self.tXXinv @ meat @ self.tXXinv
            )
  
          elif vcov_type_detail == "CRV3": 
            
            # check: is fixed effect cluster fixed effect? 
            # if not, either error or turn fixefs into dummies
            # for now: don't allow for use with fixed effects
            assert self.has_fixef == False, "CRV3 currently not supported with arbitrary fixed effects"
              
            beta_jack = np.zeros((self.G, self.k))
            tXX = np.transpose(self.X) @ self.X
            
            for ixg, g in enumerate(clustid):
              
              Xg = self.X[np.where(cluster_df == g)]
              Yg = self.Y[:,x][np.where(cluster_df == g)]
              tXgXg = np.transpose(Xg) @ Xg
  
              # jackknife regression coefficient
              beta_jack[ixg,:] = (
                np.linalg.pinv(tXX - tXgXg) @ (self.tXy[x] - np.transpose(Xg) @ Yg)
              ).flatten()
              
            beta_center = self.beta_hat[x]
            
            vcov = np.zeros((self.k, self.k))
            for ixg, g in enumerate(clustid):
              beta_centered = beta_jack[ixg,:] - beta_center
              vcov += np.outer(beta_centered, beta_centered)
            
            self.ssc.append(
              self.G / (self.G - 1)
              )
            self.vcov.append(
              self.ssc * vcov
              )

    def inference(self):
  
      self.se = []
      self.tstat = []
      self.pvalue = []
      for x in range(0, self.n_regs):
        self.se.append(
          np.sqrt(np.diagonal(self.vcov[x]))
        )
        self.tstat.append(
          self.beta_hat[x] / self.se[x]
        ) 
        self.pvalue.append(
          2*(1-norm.cdf(np.abs(self.tstat[x])))
        )
      
    def performance(self):
      
      self.r_squared = 1 - np.sum(self.u_hat ** 2) / np.sum((self.Y - np.mean(self.Y))**2)
      self.adj_r_squared = (self.N - 1) / (self.N - self.k) * self.r_squared
      

    
  
    
  
