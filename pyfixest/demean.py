import pyhdfe
import numpy as np
import pandas as pd
from formulaic import model_matrix




def _transform_fml_dict(self):

    fml_dict2 = dict()

    for fe in self.fml_dict.keys():

        fml_dict2[fe] = dict()

        for fml in self.fml_dict.get(fe):
            depvars, covars = fml.split("~")
            if fml_dict2[fe].get(depvars) is None:
                fml_dict2[fe][depvars] = [covars]
            else:
                fml_dict2[fe][depvars].append([covars])

      self.fml_dict2 = fml_dict2


def _get_na_index(self):

    na_index_dict = dict()

    for fe in self.fml_dict2.keys():

        na_index_dict[fe] = dict()

        for depvars in self.fml_dict2.get(fe).keys():

            na_index_dict[fe][depvars] = []

            for covars in self.fml_dict2.get(fe).get(depvars):

                if isinstance(covars, list):
                    covars2 = covars[0].split("+")
                else:
                    covars2 = [covars]
                if not isinstance(depvars, list):
                    depvars2 = [depvars]

                vars = depvars2 + covars2
                na_index = self.data[vars].isna().any(axis=1)
                na_index = na_index[na_index == True].index
                na_index_dict[fe][depvars].append(na_index)

    self.na_index_dict = na_index_dict


def _clean_fe(self, fval):

    fval_list = fval.split("+")

    # find interacted fixed effects via "^"
    interacted_fes = [
        x for x in fval_list if len(x.split('^')) > 1]
    regular_fes = [x for x in fval_list if len(x.split('^')) == 1]

    for x in interacted_fes:
        vars = x.split("^")
        self.data[x] = self.data[vars].apply(lambda x: '^'.join(
            x.dropna().astype(str)) if x.notna().all() else np.nan, axis=1)

    for x in regular_fes:
        self.data[x] = self.data[x].astype(str)

    fe = self.data[fval_list]
    # all fes to ints
    fe = fe.apply(lambda x: pd.factorize(x)[0])

    fe_na = fe.isna().any(axis=1)
    fe = fe.to_numpy()

    return fe, fe_na


def demean(self, fval):

    '''
    demean all regressions for a specification of fixed effects

    Args:
        fval: A specification of fixed effects. String. E.g. X4 or
              "X3 + X2"
    '''

    YX_dict = dict()
    na_dict = dict()

    if fval != "0":
        fe, fe_na = _clean_fe(self, fval)
        fe_na = list(fe_na[fe_na == True])
    else:
        fe = None
        fe_na = None

    dict2fe = self.fml_dict2.get(fval)
    na_index_dict_fe = self.na_index_dict.get(fval)

    res_fe = dict()

    na_index_old = None
    algorithm_old = None
    X_colnames_old = None
    X_demeaned_old = None

    for depvar in dict2fe.keys():

        # [(0, 'X1'), (1, ['X1+X2']), (2, ['X1+X2+X3'])]
        for c, covar in enumerate(dict2fe.get(depvar)):

            if isinstance(covar, list):
                covar2 = covar[0]
            else:
                covar2 = covar

            if isinstance(depvar, list):
                depvar2 = depvar[0]
            else:
                depvar2 = depvar

            fml = depvar2 + " ~ " + covar2

            Y, X = model_matrix(fml, self.data)
            na_index = list(set(range(0, self.data.shape[0])) - set(Y.index))

            dep_varnames = Y.columns
            co_varnames = X.columns

            Y = Y.to_numpy()
            X = X.to_numpy()

            if fe is not None:
                na_index = (na_index + fe_na)
                fe2 = np.delete(fe, na_index, axis = 0)
                # drop intercept
                X = X[:, 1:]
                co_varnames = co_varnames[1:]
                # check if variables have already been demeaned
                Y = np.delete(Y, fe_na, axis = 0)
                X = np.delete(X, fe_na, axis = 0)



            # if no demeaning has happened before, ever, or NA index for new model matrix is not the same as the previous one
            if algorithm_old is None:
                YX = np.concatenate([Y, X], axis=1)
                algorithm = pyhdfe.create(ids=fe2, residualize_method='map')
                YX_demeaned = algorithm.residualize(YX)
                YX_demeaned = pd.DataFrame(YX_demeaned)
                YX_demeaned.columns = list(dep_varnames) + list(co_varnames)
            else:
                if (list(na_index_dict_fe[depvar][c]) == na_index_old):
                # if res_depvar is not empty is NA index is the same as the previous one
                    covar_diff = list(set(co_varnames) - set(X_colnames_old))[0]
                    covar_diff_index = list(co_varnames).index(covar_diff)
                    X_covar_diff = X[:,covar_diff_index]
                    if X_covar_diff.ndim == 1:
                        X_covar_diff = X_covar_diff.reshape(len(X_covar_diff), 1)

                    # residualize needs input dimension 2
                    algorithm = algorithm_old
                    X_demeaned = algorithm.residualize(X_covar_diff)
                    YX_demeaned = np.concatenate([YX_demeaned_old, X_demeaned], axis = 1)
                    YX_demeaned = pd.DataFrame(YX_demeaned)
                    YX_demeaned.columns = list(dep_varnames) + list(co_varnames)
                else:
                    YX = np.concatenate([Y, X], axis=1)
                    algorithm = pyhdfe.create(ids=fe2, residualize_method='map')
                    YX_demeaned = algorithm.residualize(YX)
                    YX_demeaned = pd.DataFrame(YX_demeaned)
                    YX_demeaned.columns = list(dep_varnames) + list(co_varnames)


            YX_dict[fml] = YX_demeaned

            na_index_old = list(na_index)
            algorithm_old = algorithm
            X_colnames_old = co_varnames
            YX_demeaned_old = YX_demeaned

    return YX_dict







