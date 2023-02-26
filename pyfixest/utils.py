import numpy as np
from pandas import isnull
from formulaic import model_matrix



def model_matrix2(fml, data):

    fml_split = fml.split("|")
    fml_no_fixef = fml_split[0].strip()
    
    fe = None

    if len(fml_split) == 1:
        # if length = 1, then no fixed effect
        has_fixef = False
        fixef_vars = None
        Y, X = model_matrix(fml_no_fixef, data, na_action="ignore")
        depvars = Y.columns
        coefnames = X.columns
        X = np.array(X)
        Y = np.array(Y)
    else:
        has_fixef = True
        fixef_vars = fml_split[1].replace(" ", "").split("+")
        fe = data[fixef_vars]
        fe = np.array(fe)
        fe_na = np.where(np.sum(isnull(fe), axis=1) > 0)
        coefvars = fml_no_fixef.replace(" ", "").split("~")[1].split("+")
        if any(data[coefvars].dtypes == 'category'):
            Y, X = model_matrix(fml_no_fixef, data, na_action="ignore")
            depvars = Y.columns
            coefnames = X.columns
            X = np.array(X)
            Y = np.array(Y)
            # drop intercept
            X = X[:, coefnames != 'Intercept']
            coefnames = coefnames[np.where(coefnames != 'Intercept')]
        else:
            Y, X = model_matrix(fml_no_fixef + "- 1", data, na_action="ignore")
            depvars = Y.columns
            coefnames = X.columns
            X = np.array(X)
            Y = np.array(Y)

    y_na = np.where(np.sum(np.isnan(Y), axis=1) > 0)
    x_na = np.where(np.sum(np.isnan(X), axis=1) > 0)

    na_index = np.array([])
    if np.size(x_na) > 0:
        na_index = np.union1d(na_index, x_na)
    if np.size(y_na) > 0:
        na_index = np.union1d(na_index, y_na)
    if has_fixef == True:
        if np.size(fe_na) > 0:
            na_index = np.union1d(na_index, fe_na)

    na_index = na_index.astype('int')

    if len(na_index) != 0:
        Y = np.delete(Y, 0, na_index)
        X = np.delete(X, 0, na_index)
        if has_fixef == True:
            fe = np.delete(fe, 0, na_index)

    return Y, X, fe, depvars, coefnames, na_index, has_fixef, fixef_vars
