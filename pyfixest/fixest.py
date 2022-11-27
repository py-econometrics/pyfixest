import numpy as np
from pandas import isnull
from scipy.stats import norm
import pyhdfe
from formulaic import model_matrix

class fixest:
  
    def __init__(self, fml, data):
  

      self.Y, self.X, self.fe, self.depvars, self.coefnames, self.na_index, self.has_fixef, self.fixef_vars = model_matrix2(fml, data)

      self.data = data
      self.N = self.X.shape[0]
      self.k = self.X.shape[1]
      self.n_regs = self.Y.shape[1]

  
    def do_demean(self): 
      
      algorithm = pyhdfe.create(ids = self.fe, residualize_method = 'map')
      YX = np.concatenate([self.Y,self.X], axis = 1)
      residualized = algorithm.residualize(YX)
      self.Y = residualized[:, :self.n_regs]
      self.X = residualized[:, self.n_regs:]
      self.k = self.X.shape[1]

    def do_fit(self):
      
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

    def do_vcov(self, vcov):
      
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
          cluster_df = np.array(self.data[self.clustervar])
          # drop NAs
          cluster_df = np.delete(cluster_df, 0, self.na_index)
          
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
              self.ssc[x] * self.tXXinv @ meat @ self.tXXinv
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

    def do_inference(self):
  
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
      

    
 

def model_matrix2(fml, data):
  
    fml_split = fml.split("|")
    fml_no_fixef = fml_split[0].strip()

    if len(fml_split) == 1: 
      # if length = 1, then no fixed effect
      has_fixef = False
      fixef_vars = None
      Y, X = model_matrix(fml_no_fixef, data, na_action = "ignore")
      depvars = Y.columns  
      coefnames = X.columns 
      X = np.array(X)
      Y = np.array(Y)
    else: 
      has_fixef = True
      fixef_vars = fml_split[1].replace(" ", "").split("+")
      fe = data[fixef_vars]
      fe = np.array(fe)
      fe_na = np.where(np.sum(isnull(fe), axis = 1) > 0)
      coefvars = fml_no_fixef.replace(" ","").split("~")[1].split("+")
      if any(data[coefvars].dtypes == 'category'):
        Y, X = model_matrix(fml_no_fixef, data, na_action = "ignore")
        depvars = Y.columns  
        coefnames = X.columns 
        X = np.array(X)
        Y = np.array(Y)
        # drop intercept
        X = X[:,coefnames != 'Intercept']
        coefnames = coefnames[np.where(coefnames != 'Intercept')]
      else:
        Y, X = model_matrix(fml_no_fixef + "- 1", data, na_action = "ignore")
        depvars = Y.columns  
        coefnames = X.columns 
        X = np.array(X)
        Y = np.array(Y)
    
    y_na = np.where(np.sum(np.isnan(Y), axis = 1) > 0)
    x_na = np.where(np.sum(np.isnan(X), axis = 1) > 0)
  
    na_index = np.array([])
    if np.size(x_na) > 0:
        na_index = np.union1d(na_index, x_na)
    if np.size(y_na) > 0: 
        na_index = np.union1d(na_index, y_na)
    if has_fixef == True: 
      if np.size(fe_na) > 0: 
          na_index = np.union1d(na_index, fe_na)

    na_index = na_index.astype('int')
    
    Y = np.delete(Y, 0, na_index)
    X = np.delete(X, 0, na_index)
    if has_fixef == True:
      fe = np.delete(fe, 0, na_index)
    else: 
      fe = None
    
    return Y, X, fe, depvars, coefnames, na_index, has_fixef, fixef_vars

    
