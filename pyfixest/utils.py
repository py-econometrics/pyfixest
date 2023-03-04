import re
import numpy as np
import pandas as pd
from pandas import isnull
from formulaic import model_matrix


def get_fml_dict(fml):

    #fml = 'Y1 + Y2 ~ csw(X1, X2)  |a  + d'
    
    fml = "".join(fml.split())
    fml_split = fml.split('|')
    depvars, covars = fml_split[0].split("~")
    fevars = fml_split[1]

    depvars = depvars.split("+")
    covars = unpack_fml(covars)
    covars.sort(key=lambda x: 0 if isinstance(x, list) else 1)
    fevars = unpack_fml(fevars)
    fevars.sort(key=lambda x: 0 if isinstance(x, list) else 1)
    
    if isinstance(covars[0], list):
        if len(covars)>1:
            const = "+".join(covars[1:])
            covars_fml = []
            for x in covars[0]: 
                covars_fml.append(x + "+" + const)
        else: 
            covars_fml = covars[0] 
    else: 
        covars_fml = "+".join(covars)
      
    if isinstance(fevars[0], list):
        if len(fevars) > 1: 
            const = "+".join(fevars[1:])
            fevars_fml = []
            for x in fevars[0]: 
                fevars_fml.append(x + "+" + const)
        else: 
            fevars_fml = fevars[0]
    else: 
        fevars_fml = "+".join(fevars)

    if isinstance(covars_fml, str): 
        covars_fml = [covars_fml]
    if isinstance(fevars_fml, str): 
        fevars_fml = [fevars_fml]
    if isinstance(covars, str): 
        covars = [covars]
    if isinstance(fevars, str): 
        fevars = [fevars]


    
    var_dict = dict()
    for fevar in fevars_fml: 
          var_dict[fevar] = flatten(depvars) + flatten(covars)
  
    
    fml_dict = fill_formula_dict(depvars, covars_fml, fevars_fml)
      
  
    return var_dict, fml_dict


def flatten(lst):
    flattened_list = []
    for i in lst:
        if isinstance(i, list):
            flattened_list.extend(flatten(i))
        else:
            flattened_list.append(i)
    return flattened_list
  
def fill_formula_dict(depvars, covars_fml, fevars_fml): 
  
    fml_dict = dict()
    for fevar in fevars_fml:
        res = []
        for depvar in depvars:
            for covar in covars_fml:
                res.append(depvar + '~' + covar)
        fml_dict[fevar] = res
    
    return fml_dict    
    



def unpack_fml(var):

    '''
    Examples: 
        var: "a + sw(b, c)" -> ['a', ['b', 'c']]
        var = "a + csw(b, c)" -> ['a', ['b', 'b + c']]
        var = "a + csw0(b,c) + d" -> ['a', ['b', 'b + c'], 'd']

    '''
    
    res_s = []
    var_split = var.split("+")

    for x in var_split:
        
        #if isinstance(x, list) & len(x) == 1: 
        #    x = x[0]
          
        varlist, sw_type = find_sw(x)
        if sw_type == None: 
            res_s.append(x)
        else: 
            if sw_type == "sw": 
                res_s.append(varlist)
            elif sw_type == "sw0":
                res_s.append([None] + varlist)
            elif sw_type in ["csw", "csw0"]:
                varlist = ["+".join(varlist[:i+1]) for i, _ in enumerate(varlist)] 
                if sw_type == 'csw0': 
                    res_s.append([None] + varlist)
                else: 
                    res_s.append(varlist)
            else: 
                raise Exception("not supported sw type")

        
    return res_s


def find_sw(x):

    '''
    for a given string x, find all elements within 'type'
    enbracketed, e.g. 'var1, var2' in 'sw(var1, var2)'
    x = 'csw0(a, b)'
    '''

    # check for sw
    s = re.findall(r"sw\((.*?)\)", x)
    # if not empty - check if csw
    if s != []: 
        s1 = re.findall(r"csw\((.*?)\)", x)
        if s1 != []:
            return s1[0].split(","), "csw"
        else: 
            return s[0].split(","), "sw"
    else: 
        s = re.findall(r"sw0\((.*?)\)", x)
        if s != []: 
            s1 = re.findall(r"csw0\((.*?)\)", x)
            if s1 != []:
                return s1[0].split(","), "csw0"
            else: 
                return s[0].split(","), "sw0"
        else: 
            return x, None


def get_data():

    '''
    create a random example data set
    '''
    # create data
    np.random.seed(1234)
    N = 100_000
    k = 4
    G = 25
    X = np.random.normal(0, 1, N * k).reshape((N,k))
    X = pd.DataFrame(X)
    X[1] = np.random.choice(list(range(0, 50)), N, True)
    X[2] = np.random.choice(list(range(0, 100)), N, True)
    X[3] = np.random.choice(list(range(0, 100)), N, True)

    beta = np.random.normal(0,1,k)
    beta[0] = 0.005
    u = np.random.normal(0,1,N)
    Y = 1 + X @ beta + u
    cluster = np.random.choice(list(range(0,G)), N)

    Y = pd.DataFrame(Y)
    Y.rename(columns = {0:'Y'}, inplace = True)
    X = pd.DataFrame(X)

    data = pd.concat([Y, X], axis = 1)
    data.rename(columns = {0:'X1', 1:'X2', 2:'X3', 3:'X4'}, inplace = True)
    data['X4'] = data['X4'].astype('category').astype(str)
    data['X3'] = data['X3'].astype('category').astype(str)
    data['X2'] = data['X2'].astype('category').astype(str)
    data['group_id'] = cluster.astype(str)
    data['Y2'] = data.Y + np.random.normal(0, 1, N)

    return data
