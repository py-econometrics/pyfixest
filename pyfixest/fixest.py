import pyhdfe
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from typing import Any, Union, Dict, Optional, List, Tuple
from scipy.stats import norm
from formulaic import model_matrix

from pyfixest.feols import Feols
from pyfixest.fepois import Fepois
from pyfixest.feiv import Feiv
from pyfixest.FormulaParser import FixestFormulaParser
from pyfixest.ssc_utils import ssc
from pyfixest.exceptions import (
    MatrixNotFullRankError,
    MultiEstNotSupportedError,
    NotImplementedError,
)


class Fixest:
    def __init__(
        self, data: pd.DataFrame, iwls_tol: float = 1e-08, iwls_maxiter: int = 25
    ) -> None:
        """
        A class for fixed effects regression modeling.

        Args:
            data: The input pd.DataFrame for the object.
            iwls_tol: The tolerance level for the IWLS algorithm. Default is 1e-5. Only relevant for non-linear estimation strategies.
            iwls_maxiter: The maximum number of iterations for the IWLS algorithm. Default is 25. Only relevant for non-linear estimation strategies.

        Returns:
            None

        Attributes:
            data: The input pd.DataFrame for the object.
            iwls_tol: The tolerance level for the IWLS algorithm. Default is 1e-5. Only relevant for non-linear estimation strategies.
            iwls_maxiter: The maximum number of iterations for the IWLS algorithm. Default is 25. Only relevant for non-linear estimation strategies.
            all_fitted_models: A dictionary of all fitted models. The keys are the formulas used to fit the models.
        """

        self._data = None
        self._iwls_tol = None
        self._iwls_maxiter = None
        self._all_fitted_models = None

        # assert that data is a pd.DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pd.DataFrame")
        # assert that iwls_tol is a float between 0 and 1
        if not isinstance(iwls_tol, float):
            raise TypeError("iwls_tol must be a float")
        if iwls_tol < 0 or iwls_tol > 1:
            raise ValueError("iwls_tol must be between 0 and 1")
        # assert that iwls_maxiter is an integer and larger than 0
        if not isinstance(iwls_maxiter, int):
            raise TypeError("iwls_maxiter must be an integer")
        if iwls_maxiter < 1:
            raise ValueError("iwls_maxiter must be larger than 0")

        self._data = data.copy()
        # reindex: else, potential errors when pd.DataFrame.dropna()
        # -> drops indices, but formulaic model_matrix starts from 0:N...
        self._data.index = range(self._data.shape[0])
        self._iwls_tol = iwls_tol
        self._iwls_maxiter = iwls_maxiter
        self.all_fitted_models = dict()

    def _prepare_estimation(
        self,
        estimation: str,
        fml: str,
        vcov: Union[None, str, Dict[str, str]] = None,
        ssc=ssc(),
        fixef_rm: str = "none",
    ) -> None:
        """
        Utility function to prepare estimation via the `feols()` or `fepois()` methods. The function is called by both methods.
        Mostly deparses the fml string.

        Args:
            estimation: type of estimation. Either "feols" or "fepois".
            fml: A three-sided formula string using fixest formula syntax. Supported syntax includes: see `feols()` or `fepois()`.
            vcov: A string or dictionary specifying the type of variance-covariance matrix to use for inference. See `feols()` or `fepois()`.
            ssc: A dictionary specifying the type of standard errors to use for inference. See `feols()` or `fepois()`.
            fixef_rm: A string specifiny whether singleton fixed effects should be dropped.
            Options are "none" (default) and "singleton". If "singleton", singleton fixed effects are dropped.

        Returns:
            None

        Attributes:
            _fml: the provided formula string.
            _method: the estimation method. Either "feols" or "fepois".
            _is_iv: boolean indicating whether the model is an IV model.
            _fml_dict: a dictionary of deparsed formulas.
            _fml_dict_iv: a dictionary of deparsed formulas for IV models. None if no IV models. Basically, the same as
                         `_fml_dict` but with instruments.
            _ivars: a list of interaction variables. None if no interaction variables.
            _ssc_dict: a dictionary with information on small sample corrections.
            _drop_singletons: boolean indicating whether singleton fixed effects are dropped in the estimation.
            _fixef_keys: a list of fixed effects combinations.
            _drop_ref: a list of dropped reference categories for `i()` interactions.
            _split: the split variable if split estimation is used, else None.
            _splitvar: the split variable if split estimation is used, else None.
            _estimate_split_model: boolean indicating whether the split model is estimated.
            _estimate_full_model: boolean indicating whether the full model is estimated.
        """

        self._fml = fml.replace(" ", "")
        self._split = None
        self._method = estimation
        # deparse formula, at least partially

        fxst_fml = FixestFormulaParser(fml)

        if fxst_fml._is_iv:
            self._is_iv = True
        else:
            self._is_iv = False

        # add function argument to these methods for IV
        fxst_fml.get_new_fml_dict()  # fxst_fml._fml_dict_new

        self._fml_dict = fxst_fml._fml_dict_new

        if self._is_iv:
            fxst_fml.get_new_fml_dict(iv=True)  # fxst_fml._fml_dict_new
            self._fml_dict_iv = fxst_fml._fml_dict_new_iv
        else:
            self._fml_dict_iv = None

        self._ivars = fxst_fml._ivars

        self._ssc_dict = ssc
        self._drop_singletons = _drop_singletons(fixef_rm)

        # get all fixed effects combinations
        self._fixef_keys = list(self._fml_dict.keys())

        self._ivars, self._drop_ref = _clean_ivars(self._ivars, self._data)

        # currently no fsplit allowed
        fsplit = None

        (
            self._splitvar,
            _,
            self._estimate_split_model,
            self._estimate_full_model,
        ) = _prepare_split_estimation(self._split, fsplit, self._data, self._fml_dict)

    def feols(
        self,
        fml: str,
        vcov: Optional[Union[str, Dict[str, str]]] = None,
        ssc=ssc(),
        fixef_rm: str = "none",
        weights=Optional,
    ) -> None:
        """
        Method for fixed effects regression modeling using the PyHDFE package for projecting out fixed effects.
        Args:
            fml (str): A three-sided formula string using fixest formula syntax. Supported syntax includes:
                The syntax is as follows: "Y ~ X1 + X2 | FE1 + FE2 | X1 ~ Z1" where:

                Y: Dependent variable
                X1, X2: Independent variables
                FE1, FE2: Fixed effects
                Z1, Z2: Instruments
                |: Separates left-hand side, fixed effects, and instruments

                If no fixed effects and instruments are specified, the formula can be simplified to "Y ~ X1 + X2".
                If no instruments are specified, the formula can be simplified to "Y ~ X1 + X2 | FE1 + FE2".
                If no fixed effects are specified but instruments are specified, the formula can be simplified to "Y ~ X1 + X2 | X1 ~ Z1".

                Supported multiple estimation syntax includes:

                Stepwise regressions (sw, sw0)
                Cumulative stepwise regression (csw, csw0)
                Multiple dependent variables (Y1 + Y2 ~ X)

                Other special syntax includes:
                i() for interaction of a categorical and non-categorical variable (e.g. "i(X1,X2)" for interaction between X1 and X2).
                Using i() is required to use with some custom methods, e.g. iplot().
                ^ for interacted fixed effects (e.g. "fe1^fe2" for interaction between fe1 and fe2)

                All other parts of the formula must be compatible with formula parsing via the formulaic module.
                You can use formulaic functionaloty such as "C", "I", ":",, "*", "np.log", "np.power", etc.

            vcov (Union(str, dict)): A string or dictionary specifying the type of variance-covariance matrix to use for inference.
                If a string, it can be one of "iid", "hetero", "HC1", "HC2", "HC3".
                If a dictionary, it should have the format dict("CRV1":"clustervar") for CRV1 inference or dict(CRV3":"clustervar") for CRV3 inference.
            fixef_rm: A string specifiny whether singleton fixed effects should be dropped. Options are "none" (default) and "singleton". If "singleton", singleton fixed effects are dropped.
        Returns:
            None
        Examples:
            Standard formula:
                fml = 'Y ~ X1 + X2'
                fixest_model = Fixest(data=data).feols(fml, vcov='iid')
            With fixed effects:
                fml = 'Y ~ X1 + X2 | fe1 + fe2'
            With interacted fixed effects:
                fml = 'Y ~ X1 + X2 | fe1^fe2'
            Multiple dependent variables:
                fml = 'Y1 + Y2 ~ X1 + X2'
            Stepwise regressions (sw and sw0):
                fml = 'Y1 + Y2 ~ sw(X1, X2, X3)'
            Cumulative stepwise regressions (csw and csw0):
                fml = 'Y1 + Y2 ~ csw(X1, X2, X3) '
            Combinations:
                fml = 'Y1 + Y2 ~ csw(X1, X2, X3) | sw(X4, X5) + X6'
            With instruments:
                fml = 'Y ~ X1 + X2 | X1 ~ Z1'
            With instruments and fixed effects:
                fml = 'Y ~ X1 + X2 | X1 ~ Z1  | fe1 + fe2'

        Attributes:
            - attributes set via _prepare_estimation():
                _fml: the provided formula string.
                _method: the estimation method. Either "feols" or "fepois".
                _is_iv: boolean indicating whether the model is an IV model.
                _fml_dict: a dictionary of deparsed formulas.
                _fml_dict_iv: a dictionary of deparsed formulas for IV models. None if no IV models. Basically, the same as
                                `_fml_dict` but with instruments.
                _ivars: a list of interaction variables. None if no interaction variables.
                _ssc_dict: a dictionary with information on small sample corrections.
                _drop_singletons: boolean indicating whether singleton fixed effects are dropped in the estimation.
                _fixef_keys: a list of fixed effects combinations.
                _drop_ref: a list of dropped reference categories for `i()` interactions.
                _split: the split variable if split estimation is used, else None.
                _splitvar: the split variable if split estimation is used, else None.
                _estimate_split_model: boolean indicating whether the split model is estimated.
                _estimate_full_model: boolean indicating whether the full model is estimated.
            - attributes set via _model_matrix_fixest():
                icovars: a list of interaction variables. None if no interaction variables via `i()` provided.
            - attributes set via _estimate_all_models():

            - attributes set via _is_multiple_estimation():
                is_fixef_multi: boolean indicating whether multiple regression models will be estimated

        """

        self._prepare_estimation("feols", fml, vcov, ssc, fixef_rm)

        # demean all models: based on fixed effects x split x missing value combinations
        self._estimate_all_models(vcov, self._fixef_keys)

        # create self._is_fixef_multi flag
        self._is_multiple_estimation()

        if self._is_fixef_multi and self._is_iv:
            raise MultiEstNotSupportedError(
                "Multiple Estimations is currently not supported with IV."
                "This is mostly due to insufficient testing and will be possible with the next release of PyFixest."
            )

        return self

    def fepois(
        self,
        fml: str,
        vcov: Optional[Union[str, Dict[str, str]]] = None,
        ssc=ssc(),
        fixef_rm: str = "none",
    ) -> None:
        """
        Method for Estimation of Poisson Regression with high-dimensional fixed effects. See `feols()` for more details.
        """

        self._prepare_estimation(
            estimation="fepois", fml=fml, vcov=vcov, ssc=ssc, fixef_rm=fixef_rm
        )

        if self._is_iv:
            raise NotImplementedError(
                "IV Estimation is not supported for Poisson Regression"
            )

        self._estimate_all_models(vcov, self._fixef_keys)

        # create self._is_fixef_multi flag
        self._is_multiple_estimation()

        return self

    def _clean_fe(
        self, data: pd.DataFrame, fval: str
    ) -> Tuple[pd.DataFrame, List[int]]:
        """
        Clean and transform fixed effects in a DataFrame.

        This is a helper function used in `_model_matrix_fixest()`. The function converts
        the fixed effects to integers and marks fixed effects with NaNs. It's important
        to note that NaNs are not removed at this stage; this is done in `_model_matrix_fixest()`.

        Args:
            data (pd.DataFrame): The input DataFrame containing the data.
            fval (str): A string describing the fixed effects, e.g., "fe1 + fe2".

        Returns:
            Tuple[pd.DataFrame, List[int]]: A tuple containing two items:
                - fe (pd.DataFrame): The DataFrame with cleaned fixed effects. NaNs are
                present in this DataFrame.
                - fe_na (List[int]): A list of columns in 'fe' that contain NaN values.
        """

        fval_list = fval.split("+")

        # find interacted fixed effects via "^"
        interacted_fes = [x for x in fval_list if len(x.split("^")) > 1]

        for x in interacted_fes:
            vars = x.split("^")
            data[x] = data[vars].apply(
                lambda x: "^".join(x.dropna().astype(str))
                if x.notna().all()
                else np.nan,
                axis=1,
            )

        fe = data[fval_list]

        for x in fe.columns:
            if fe[x].dtype != "category":
                if len(fe[x].unique()) == fe.shape[0]:
                    raise ValueError(
                        f"Fixed effect {x} has only unique values. "
                        "This is not allowed."
                    )

        fe_na = fe.isna().any(axis=1)
        fe = fe.apply(lambda x: pd.factorize(x)[0])
        fe_na = fe_na[fe_na].index.tolist()

        return fe, fe_na

    def _model_matrix_fixest(
        self,
        fml: str,
        data: pd.DataFrame,
        weights: Optional[str] = None
    ) -> Tuple[
        pd.DataFrame,  # Y
        pd.DataFrame,  # X
        Optional[pd.DataFrame],  # I
        Optional[pd.DataFrame],  # fe
        np.ndarray,  # na_index
        np.ndarray,  # fe_na
        str,  # na_index_str
        Optional[List[str]],  # z_names
        Optional[str],  # weights
        bool,  # has_weights
    ]:
        """
        Create model matrices for fixed effects estimation.

        This function preprocesses the data and then calls `formulaic.model_matrix()`
        to create the model matrices.

        Args:
            fml (str): A two-sided formula string using fixest formula syntax.
            weights (str or None): Weights as a string if provided, or None if no weights, e.g., "weights".

        Returns:
            Tuple[
                pd.DataFrame,  # Y
                pd.DataFrame,  # X
                Optional[pd.DataFrame],  # I
                Optional[pd.DataFrame],  # fe
                np.array,  # na_index
                np.array,  # fe_na
                str,  # na_index_str
                Optional[List[str]],  # z_names
                Optional[str],  # weights
                bool  # has_weights
            ]: A tuple of the following elements:
                - Y: A DataFrame of the dependent variable.
                - X: A DataFrame of the covariates. If `combine = True`, contains covariates and fixed effects as dummies.
                - I: A DataFrame of the Instruments, None if no IV.
                - fe: A DataFrame of the fixed effects, None if no fixed effects specified. Only applicable if `combine = False`.
                - na_index: An array with indices of dropped columns.
                - fe_na: An array with indices of dropped columns due to fixed effect singletons or NaNs in the fixed effects.
                - na_index_str: na_index, but as a comma-separated string. Used for caching of demeaned variables.
                - z_names: Names of all covariates, minus the endogenous variables, plus the instruments. None if no IV.
                - weights: Weights as a string if provided, or None if no weights, e.g., "weights".
                - has_weights: A boolean indicating whether weights are used.

        Attributes:
            list or None: icovars - A list of interaction variables. None if no interaction variables via `i()` provided.
        """

        _is_iv = self._is_iv
        _ivars = self._ivars
        _drop_ref = self._drop_ref

        # step 1: deparse formula
        fml_parts = fml.split("|")
        depvar, covar = fml_parts[0].split("~")

        if len(fml_parts) == 3:
            fval, fml_iv = fml_parts[1], fml_parts[2]
        elif len(fml_parts) == 2:
            if _is_iv:
                fval, fml_iv = "0", fml_parts[1]
            else:
                fval, fml_iv = fml_parts[1], None
        else:
            fval = "0"
            fml_iv = None

        if _is_iv:
            endogvar, instruments = fml_iv.split("~")
        else:
            endogvar, instruments = None, None

        # step 2: create formulas
        fml_exog = depvar + " ~ " + covar
        if _is_iv:
            fml_iv_full = fml_iv + "+" + covar + "-" + endogvar

        # clean fixed effects
        if fval != "0":
            fe, fe_na = self._clean_fe(data, fval)
            #fml_exog += " | " + fval
        else:
            fe = None
            fe_na = None
        # fml_iv already created

        Y, X = model_matrix(fml_exog, data)
        if _is_iv:
            endogvar, Z = model_matrix(fml_iv_full, data)
        else:
            endogvar, Z = None, None

        Y, X, endogvar, Z = [pd.DataFrame(x) if x is not None else x for x in [Y, X, endogvar, Z]]

        # check if Y, endogvar have dimension (N, 1) - else they are non-numeric
        if Y.shape[1] > 1:
            raise TypeError(f"The dependent variable must be numeric, but it is of type {data[depvar].dtype}.")
        if endogvar is not None:
            if endogvar.shape[1] > 1:
                raise TypeError(f"The endogenous variable must be numeric, but it is of type {data[endogvar].dtype}.")

        # step 3: catch NaNs (before converting to numpy arrays)
        na_index_stage2 = list(set(data.index) - set(Y.index))

        if _is_iv:
            na_index_stage1 = list(set(data.index) - set(Z.index))
            diff1 = list(set(na_index_stage1) - set(na_index_stage2))
            diff2 = list(set(na_index_stage2) - set(na_index_stage1))
            if diff1:
                Y = Y.drop(diff1, axis=0)
                X = X.drop(diff1, axis=0)
            if diff2:
                Z = Z.drop(diff2, axis=0)
                endogvar = endogvar.drop(diff2, axis=0)
            na_index = list(set(na_index_stage1 + na_index_stage2))
        else:
            na_index = na_index_stage2

        # drop variables before collecting variable names
        if _ivars is not None:
            if _drop_ref is not None:
                X = X.drop(_drop_ref, axis=1)
                if _is_iv:
                    Z = Z.drop(_drop_ref, axis=1)

        if _ivars is not None:
            x_names = X.columns.tolist()
            self._icovars = [
                s for s in x_names if s.startswith(_ivars[0]) and s.endswith(_ivars[1])
            ]
        else:
            self._icovars = None

        if fe is not None:
            fe = fe.drop(na_index, axis=0)
            # drop intercept
            X = X.drop("Intercept", axis=1)
            #x_names.remove("Intercept")
            if _is_iv:
                Z = Z.drop("Intercept", axis=1)
            #    z_names.remove("Intercept")

            # drop NaNs in fixed effects (not yet dropped via na_index)
            fe_na_remaining = list(set(fe_na) - set(na_index))
            if fe_na_remaining:
                Y = Y.drop(fe_na_remaining, axis=0)
                X = X.drop(fe_na_remaining, axis=0)
                fe = fe.drop(fe_na_remaining, axis=0)
                if _is_iv:
                    Z = Z.drop(fe_na_remaining, axis=0)
                    endogvar = endogvar.drop(fe_na_remaining, axis=0)
                na_index += fe_na_remaining
                na_index = list(set(na_index))

        N = X.shape[0]

        na_index_str = ",".join(str(x) for x in na_index)

        return Y, X, fe, endogvar, Z, na_index, na_index_str

    def _demean_model(
        self,
        Y: pd.DataFrame,
        X: pd.DataFrame,
        fe: Optional[pd.DataFrame],
        weights: Optional[np.ndarray],
        lookup_demeaned_data: Dict[str, Any],
        na_index_str: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Demeans a single regression model.

        If the model has fixed effects, the fixed effects are demeaned using the PyHDFE package.
        Prior to demeaning, the function checks if some of the variables have already been demeaned and uses values
        from the cache `lookup_demeaned_data` if possible. If the model has no fixed effects, the function does not demean the data.

        Args:
            Y (pd.DataFrame): A DataFrame of the dependent variable.
            X (pd.DataFrame): A DataFrame of the covariates.
            fe (pd.DataFrame or None): A DataFrame of the fixed effects. None if no fixed effects specified.
            weights (np.ndarray or None): A numpy array of weights. None if no weights.
            lookup_demeaned_data (Dict[str, Any]): A dictionary with keys for each fixed effects combination and
                potentially values of demeaned data frames. The function checks this dictionary to see if some of
                the variables have already been demeaned.
            na_index_str (str): A string with indices of dropped columns. Used for caching of demeaned variables.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]: A tuple of the following elements:
                - Yd (pd.DataFrame): A DataFrame of the demeaned dependent variable.
                - Xd (pd.DataFrame): A DataFrame of the demeaned covariates.
                - Id (pd.DataFrame or None): A DataFrame of the demeaned Instruments. None if no IV.
        """

        #if val == 1:
        #    import pdb
        #    pdb.set_trace()

        _drop_singletons = self._drop_singletons

        YX = pd.concat([Y, X], axis=1)

        yx_names = YX.columns
        YX = YX.to_numpy()

        if fe is not None:
            # check if looked dict has data for na_index
            if lookup_demeaned_data.get(na_index_str) is not None:
                # get data out of lookup table: list of [algo, data]
                algorithm, YX_demeaned_old = lookup_demeaned_data.get(na_index_str)

                # get not yet demeaned covariates
                var_diff_names = list(set(yx_names) - set(YX_demeaned_old.columns))

                # if some variables still need to be demeaned
                if var_diff_names:
                    #var_diff_names = var_diff_names

                    yx_names_list = list(yx_names)
                    var_diff_index = [yx_names_list.index(item) for item in var_diff_names]
                    #var_diff_index = list(yx_names).index(var_diff_names)
                    var_diff = YX[:, var_diff_index]
                    if var_diff.ndim == 1:
                        var_diff = var_diff.reshape(len(var_diff), 1)

                    YX_demean_new = algorithm.residualize(var_diff)
                    YX_demeaned = np.concatenate(
                        [YX_demeaned_old, YX_demean_new], axis=1
                    )
                    YX_demeaned = pd.DataFrame(YX_demeaned)

                    # check if var_diff_names is a list
                    if isinstance(var_diff_names, str):
                        var_diff_names = [var_diff_names]

                    YX_demeaned.columns = list(YX_demeaned_old.columns) + var_diff_names

                else:
                    # all variables already demeaned
                    YX_demeaned = YX_demeaned_old[yx_names]

            else:
                # not data demeaned yet for NA combination
                algorithm = pyhdfe.create(
                    ids=fe,
                    residualize_method="map",
                    drop_singletons=_drop_singletons,
                    # weights=weights
                )

                if (
                    _drop_singletons == True
                    and algorithm.singletons != 0
                    and algorithm.singletons is not None
                ):
                    print(
                        algorithm.singletons,
                        "columns are dropped due to singleton fixed effects.",
                    )
                    dropped_singleton_indices = np.where(algorithm._singleton_indices)[
                        0
                    ].tolist()
                    na_index += dropped_singleton_indices

                YX_demeaned = algorithm.residualize(YX)
                YX_demeaned = pd.DataFrame(YX_demeaned)

                YX_demeaned.columns = yx_names

            lookup_demeaned_data[na_index_str] = [algorithm, YX_demeaned]

        else:
            # nothing to demean here
            pass

            YX_demeaned = pd.DataFrame(YX)
            YX_demeaned.columns = yx_names

        # get demeaned Y, X (if no fixef, equal to Y, X, I)
        Yd = YX_demeaned[Y.columns]
        Xd = YX_demeaned[X.columns]

        return Yd, Xd

    def _estimate_all_models(
        self, vcov: Union[str, Dict[str, str]], fixef_keys: List[str]
    ) -> None:
        """
        Estimate multiple regression models.

        Args:
            vcov (Union[str, Dict[str, str]]): A string or dictionary specifying the type of variance-covariance
                matrix to use for inference.
                - If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3".
                - If a dictionary, it should have the format {"CRV1": "clustervar"} for CRV1 inference
                  or {"CRV3": "clustervar"} for CRV3 inference.
            fixef_keys (List[str]): A list of fixed effects combinations.

        Returns:
            None

        Attributes:
            all_fitted_models (Dict[str, Any]): A dictionary of all fitted models. The keys are the formulas
                used to fit the models.
        """
        # pdb.set_trace()

        _estimate_full_model = self._estimate_full_model
        # _estimate_split_model = self._estimate_split_model
        _fml_dict = self._fml_dict
        _is_iv = self._is_iv
        _data = self._data
        _method = self._method
        _ivars = self._ivars
        # _drop_ref = self._drop_ref
        _drop_singletons = self._drop_singletons
        _ssc_dict = self._ssc_dict
        _iwls_maxiter = self._iwls_maxiter
        _iwls_tol = self._iwls_tol
        # _icovars = self._icovars

        if _estimate_full_model:
            for _, fval in enumerate(fixef_keys):
                dict2fe = _fml_dict.get(fval)

                # dictionary to cache demeaned data with index: na_index_str,
                # only relevant for `.feols()`
                lookup_demeaned_data = dict()

                # loop over both dictfe and dictfe_iv (if the latter is not None)
                for depvar in dict2fe.keys():
                    for _, fml_linear in enumerate(dict2fe.get(depvar)):
                        if isinstance(fml_linear, list):
                            fml_linear = fml_linear[0]

                        covar = fml_linear.split("~")[1]

                        if _is_iv:
                            dict2fe_iv = self._fml_dict_iv.get(fval)
                            instruments2 = dict2fe_iv.get(depvar)[0].split("~")[1]
                            endogvar_list = list(set(covar.split("+")) - set(instruments2.split("+")))
                            instrument_list = list(set(instruments2.split("+")) - set(covar.split("+")))
                            endogvars = endogvar_list[0]
                            instruments = "+".join(instrument_list)
                        else:
                            endogvars = None
                            instruments = None


                        fml = get_fml(depvar, covar, fval, endogvars, instruments)

                        # get Y, X, Z, fe, NA indices for model
                        Y, X, fe, endogvar, Z, na_index, na_index_str = self._model_matrix_fixest(fml = fml, data = self._data)
                        weights = np.ones((Y.shape[0], 1))

                        y_names = Y.columns.tolist()
                        x_names = X.columns.tolist()
                        if _is_iv:
                            z_names = Z.columns.tolist()
                            endogvar_names = endogvar.columns.tolist()
                        else:
                            z_names = None
                            endogvar_names = None

                        if _method == "feols":

                            # demean Y, X, Z, if not already done in previous estimation


                            Yd, Xd = self._demean_model(
                                Y, X, fe, weights, lookup_demeaned_data, na_index_str
                            )

                            if _is_iv:
                                endogvard, Zd = self._demean_model(
                                    endogvar, Z, fe, weights, lookup_demeaned_data, na_index_str
                                )
                            else:
                                endogvard, Zd = None, None



                            if not _is_iv:
                                Zd = Xd

                            Yd, Xd, Zd, endogvard = [x.to_numpy() if x is not None else x for x in [Yd, Xd, Zd, endogvard]]

                            has_weights = False
                            if has_weights:
                                w = np.sqrt(weights.to_numpy())
                                Yd *= np.sqrt(w)
                                Zd *= np.sqrt(w)
                                Xd *= np.sqrt(w)

                            # check for multicollinearity
                            _multicollinearity_checks(Xd, Zd, _ivars, fml)

                            if _is_iv:
                                FIT = Feiv(Y=Yd, X=Xd, Z=Zd, weights=weights)
                            else:
                                # initiate OLS class
                                FIT = Feols(Y=Yd, X=Xd, weights=weights)

                            FIT.get_fit()

                        elif _method == "fepois":
                            # check for separation and drop separated variables
                            # Y, X, fe, na_index = self._separation()

                            Y, X = [x.to_numpy() for x in [Y, X]]
                            N = X.shape[0]

                            if fe is not None:
                                fe = fe.to_numpy()
                                if fe.ndim == 1:
                                    fe = fe.reshape((N, 1))

                            # check for multicollinearity
                            _multicollinearity_checks(X, X, _ivars, fml)

                            # initiate OLS class
                            FIT = Fepois(
                                Y=Y,
                                X=X,
                                fe=fe,
                                weights=weights,
                                drop_singletons=_drop_singletons,
                                maxiter=_iwls_maxiter,
                                tol=_iwls_tol,
                            )

                            FIT._is_iv = False
                            FIT.get_fit()

                            FIT.na_index = na_index
                            FIT.n_separation_na = None
                            if FIT.separation_na:
                                FIT.na_index += FIT.separation_na
                                FIT.n_separation_na = len(FIT.separation_na)

                        else:
                            raise ValueError(
                                "Estimation method not supported. Please use 'feols' or 'fepois'."
                            )

                        # some bookkeeping
                        FIT._fml = fml
                        FIT._ssc_dict = _ssc_dict
                        # FIT._na_index = na_index
                        # data never makes it to Feols() class. needed for ex post
                        # clustered vcov estimation when clustervar not in model params
                        FIT._data = _data.iloc[~_data.index.isin(na_index)]
                        if fval != "0":
                            FIT._has_fixef = True
                            FIT._fixef = fval
                        else:
                            FIT._has_fixef = False
                            FIT._fixef = None
                        # FEOLS.split_log = x

                        # inference
                        vcov_type = _get_vcov_type(vcov, fval)
                        FIT._vcov_log = vcov_type
                        FIT.get_vcov(vcov=vcov_type)
                        FIT.get_inference()

                        # other regression stats
                        FIT.get_performance()

                        FIT._coefnames = x_names
                        if self._icovars is not None:
                            FIT._icovars = self._icovars
                        else:
                            FIT._icovars = None

                        # store fitted model
                        self.all_fitted_models[fml] = FIT

    def _is_multiple_estimation(self) -> None:
        """
        Helper method to check if multiple regression models will be estimated.

        Args:
            None

        Returns:
            None

        Attributes:
            is_fixef_multi (bool): A boolean indicating whether multiple regression models will be estimated.
        """

        self._is_fixef_multi = False
        if len(self._fml_dict.keys()) > 1:
            self._is_fixef_multi = True
        elif len(self._fml_dict.keys()) == 1:
            first_key = next(iter(self._fml_dict))
            if len(self._fml_dict[first_key]) > 1:
                self._is_fixef_multi = True

    def vcov(self, vcov: Union[str, Dict[str, str]]) -> None:
        """
        Update regression inference "on the fly".

        By calling vcov() on a "Fixest" object, all inference procedures applied
        to the "Fixest" object are replaced with the variance-covariance matrix specified via the method.

        Args:
            vcov (Union[str, Dict[str, str]]): A string or dictionary specifying the type of variance-covariance
                matrix to use for inference.
                - If a string, can be one of "iid", "hetero", "HC1", "HC2", "HC3".
                - If a dictionary, it should have the format {"CRV1": "clustervar"} for CRV1 inference
                  or {"CRV3": "clustervar"} for CRV3 inference.

        Returns:
            None
        """

        for model in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[model]
            fxst._vcov_log = vcov

            fxst.get_vcov(vcov=vcov)
            fxst.get_inference()

        return self

    def tidy(self) -> pd.DataFrame:
        """
        Returns the results of an estimation using `feols()` as a tidy Pandas DataFrame.
        Returns:
            pd.DataFrame or str
                A tidy DataFrame with the following columns:
                - fml: the formula used to generate the results
                - Coefficient: the names of the coefficients
                - Estimate: the estimated coefficients
                - Std. Error: the standard errors of the estimated coefficients
                - t value: the t-values of the estimated coefficients
                - Pr(>|t|): the p-values of the estimated coefficients
                - 2.5 %: the lower bound of the 95% confidence interval
                - 97.5 %: the upper bound of the 95% confidence interval
                If `type` is set to "markdown", the resulting DataFrame will be returned as a
                markdown-formatted string with three decimal places.
        """

        res = []
        for x in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[x]
            df = fxst.tidy().reset_index()
            df["fml"] = fxst._fml
            res.append(df)

        res = pd.concat(res, axis=0).set_index(["fml", "Coefficient"])

        return res

    def etable(self, digits: int = 3) -> None:
        return self.tidy().T.round(digits)

    def summary(self, digits: int = 3) -> None:
        """
        Prints a summary of the feols() estimation results for each estimated model.

        For each model, the method prints a header indicating the fixed-effects and the
        dependent variable, followed by a table of coefficient estimates with standard
        errors, t-values, and p-values.

        Args:
            digits (int, optional): The number of decimal places to round the summary statistics to. Default is 3.

        Returns:
            None
        """

        for x in list(self.all_fitted_models.keys()):
            split = x.split("|")
            if len(split) > 1:
                fe = split[1]
            else:
                fe = None
            depvar = split[0].split("~")[0]
            fxst = self.all_fitted_models[x]

            df = fxst.tidy().round(digits)

            if fxst._method == "feols":
                if fxst._is_iv:
                    estimation_method = "IV"
                else:
                    estimation_method = "OLS"
            else:
                estimation_method = "Poisson"

            print("###")
            print("")
            print("Model: ", estimation_method)
            print("Dep. var.: ", depvar)
            if fe is not None:
                print("Fixed effects: ", fe)
            # if fxst.split_log is not None:
            #    print('Split. var: ', self._split + ":" + fxst.split_log)
            print("Inference: ", fxst._vcov_log)
            print("Observations: ", fxst.N)
            print("")
            print(df.to_markdown(floatfmt="." + str(digits) + "f"))
            print("---")
            if fxst._method == "feols":
                if not fxst._is_iv:
                    print(
                        f"RMSE: {np.round(fxst.rmse, digits)}  Adj. R2: {np.round(fxst.adj_r2, digits)}  Adj. R2 Within: {np.round(fxst.adj_r2_within, digits)}"
                    )
            elif fxst._method == "fepois":
                print(f"Deviance: {np.round(fxst.deviance[0], digits)}")
            else:
                pass

    def coef(self) -> pd.DataFrame:
        """
        Obtain the coefficients of the fitted models.
        Returns:
            A pd.DataFrame with coefficient names and Estimates. The key indicates which models the estimated statistic derives from.
        """
        return self.tidy()["Estimate"]

    def se(self) -> pd.DataFrame:
        """
        Obtain the standard errors of the fitted models.

        Returns:
            A pd.DataFrame with coefficient names and standard error estimates. The key indicates which models the estimated statistic derives from.

        """
        return self.tidy()["Std. Error"]

    def tstat(self) -> pd.DataFrame:
        """
        Obtain the t-statistics of the fitted models.

         Returns:
            A pd.DataFrame with coefficient names and estimated t-statistics. The key indicates which models the estimated statistic derives from.

        """
        return self.tidy()["t value"]

    def pvalue(self) -> pd.DataFrame:
        """
        Obtain the p-values of the fitted models.

        Returns:
            A pd.DataFrame with coefficient names and p-values. The key indicates which models the estimated statistic derives from.

        """
        return self.tidy()["Pr(>|t|)"]

    def confint(self) -> pd.DataFrame:
        """'
        Obtain confidence intervals for the fitted models.

        Returns:
            A pd.DataFrame with coefficient names and confidence intervals. The key indicates which models the estimated statistic derives from.
        """

        return self.tidy()[["2.5 %", "97.5 %"]]

    def iplot(
        self,
        alpha: float = 0.05,
        figsize: tuple = (10, 10),
        yintercept: Union[int, str, None] = None,
        xintercept: Union[int, str, None] = None,
        rotate_xticks: int = 0,
    ) -> None:
        """
        Plot model coefficients with confidence intervals for variable interactions specified via the `i()` syntax.

        Args:
            alpha (float, optional): The significance level for the confidence intervals. Default is 0.05.
            figsize (tuple, optional): The size of the figure. Default is (10, 10).
            yintercept (Union[int, str, None], optional): The value at which to draw a horizontal line.
            xintercept (Union[int, str, None], optional): The value at which to draw a vertical line.
            rotate_xticks (int, optional): The rotation angle for x-axis tick labels. Default is 0.

        Returns:
            None
        """

        ivars = self._icovars

        if ivars is None:
            raise ValueError(
                "The estimated models did not have ivars / 'i()' model syntax."
                "In consequence, the '.iplot()' method is not supported."
            )

        if "Intercept" in ivars:
            ivars.remove("Intercept")

        df = self.tidy().reset_index()

        df = df[df.Coefficient.isin(ivars)]
        models = df.fml.unique()

        _coefplot(
            models=models,
            figsize=figsize,
            alpha=alpha,
            yintercept=yintercept,
            xintercept=xintercept,
            df=df,
            is_iplot=True,
        )

    def coefplot(
        self,
        alpha: float = 0.05,
        figsize: tuple = (5, 2),
        yintercept: int = 0,
        figtitle: Optional[str] = None,
        figtext: Optional[str] = None,
        rotate_xticks: int = 0,
    ) -> None:
        """
        Plot estimation results. The plot() method is only defined for single regressions.
        Args:
            alpha (float): the significance level for the confidence intervals. Default is 0.05.
            figsize (tuple): the size of the figure. Default is (5, 2).
            yintercept (float): the value of the y-intercept. Default is 0.
            figtitle (str, optional): The title of the figure. Default is None.
            figtext (str, optional): The text at the bottom of the figure. Default is None.
        Returns:
            None
        """

        df = self.tidy().reset_index()
        models = df.fml.unique()

        _coefplot(
            models=models,
            figsize=figsize,
            alpha=alpha,
            yintercept=yintercept,
            xintercept=None,
            df=df,
            is_iplot=False,
            rotate_xticks=rotate_xticks,
        )

    def wildboottest(
        self,
        B: int,
        param: Optional[str] = None,
        weights_type: str = "rademacher",
        impose_null: bool = True,
        bootstrap_type: str = "11",
        seed: Optional[str] = None,
        adj: bool = True,
        cluster_adj: bool = True,
    ) -> pd.DataFrame:
        """
        Run a wild cluster bootstrap for all regressions in the Fixest object.

        Args:
            B (int): The number of bootstrap iterations to run.
            param (Union[str, None], optional): A string of length one, containing the test parameter of interest. Default is None.
            weights_type (str, optional): The type of bootstrap weights. Either 'rademacher', 'mammen', 'webb', or 'normal'.
                Default is 'rademacher'.
            impose_null (bool, optional): Should the null hypothesis be imposed on the bootstrap dgp, or not?
                Default is True.
            bootstrap_type (str, optional): A string of length one. Allows choosing the bootstrap type
                to be run. Either '11', '31', '13', or '33'. Default is '11'.
            seed (Union[str, None], optional): Option to provide a random seed. Default is None.
            adj (bool, optional): Whether to adjust the original coefficients with the bootstrap distribution.
                Default is True.
            cluster_adj (bool, optional): Whether to adjust standard errors for clustering in the bootstrap.
                Default is True.

        Returns:
            A pd.DataFrame with bootstrapped t-statistic and p-value. The index indicates which model the estimated
            statistic derives from.
        """

        res = []
        for x in list(self.all_fitted_models.keys()):
            fxst = self.all_fitted_models[x]

            if hasattr(fxst, "clustervar"):
                cluster = fxst.clustervar
            else:
                cluster = None

            boot_res = fxst.get_wildboottest(
                B,
                cluster,
                param,
                weights_type,
                impose_null,
                bootstrap_type,
                seed,
                adj,
                cluster_adj,
            )

            pvalue = boot_res["pvalue"]
            tstat = boot_res["statistic"]

            res.append(
                pd.Series(
                    {"fml": x, "param": param, "t value": tstat, "Pr(>|t|)": pvalue}
                )
            )

        res = pd.concat(res, axis=1).T.set_index("fml")

        return res

    def fetch_model(self, i: Union[int, str]) -> Union[Feols, Fepois]:
        """
        Utility method to fetch a model of class Feols from the Fixest class.
        Args:
            i (int or str): The index of the model to fetch.
        Returns:
            A Feols object.
        """

        if isinstance(i, str):
            i = int(i)

        keys = list(self.all_fitted_models.keys())
        if i >= len(keys):
            raise IndexError(f"Index {i} is larger than the number of fitted models.")
        key = keys[i]
        print("Model: ", key)
        model = self.all_fitted_models[key]
        return model


def _coefplot(
    models: List,
    df: pd.DataFrame,
    figsize: Tuple[int, int],
    alpha: float,
    yintercept: Optional[int] = None,
    xintercept: Optional[int] = None,
    is_iplot: bool = False,
    rotate_xticks: float = 0,
) -> None:
    """
    Plot model coefficients with confidence intervals.
    Args:
        models (list): A list of fitted models indices.
        figsize (tuple): The size of the figure.
        alpha (float): The significance level for the confidence intervals.
        yintercept (int or None): The value at which to draw a horizontal line on the plot.
        xintercept (int or None): The value at which to draw a vertical line on the plot.
        df (pandas.DataFrame): The dataframe containing the data used for the model fitting.
        is_iplot (bool): If True, plot variable interactions specified via the `i()` syntax.
        rotate_xticks (float): The angle in degrees to rotate the xticks labels. Default is 0 (no rotation).
    Returns:
    None
    """

    if len(models) > 1:
        fig, ax = plt.subplots(
            len(models), gridspec_kw={"hspace": 0.5}, figsize=figsize
        )

        for x, model in enumerate(models):
            df_model = df.reset_index().set_index("fml").xs(model)
            coef = df_model["Estimate"]
            conf_l = coef - df_model["Std. Error"] * norm.ppf(1 - alpha / 2)
            conf_u = coef + df_model["Std. Error"] * norm.ppf(1 - alpha / 2)
            coefnames = df_model["Coefficient"].values.tolist()

            # could be moved out of the for loop, as the same ivars for all
            # models.

            if is_iplot == True:
                fig.suptitle("iplot")
                coefnames = [
                    (i)
                    for string in coefnames
                    for i in re.findall(r"\[T\.([\d\.\-]+)\]", string)
                ]

            ax[x].scatter(coefnames, coef, color="b", alpha=0.8)
            ax[x].scatter(coefnames, conf_u, color="b", alpha=0.8, marker="_", s=100)
            ax[x].scatter(coefnames, conf_l, color="b", alpha=0.8, marker="_", s=100)
            ax[x].vlines(coefnames, ymin=conf_l, ymax=conf_u, color="b", alpha=0.8)
            if yintercept is not None:
                ax[x].axhline(yintercept, color="red", linestyle="--", alpha=0.5)
            if xintercept is not None:
                ax[x].axvline(xintercept, color="red", linestyle="--", alpha=0.5)
            ax[x].set_ylabel("Coefficients")
            ax[x].set_title(model)
            ax[x].tick_params(axis="x", rotation=rotate_xticks)

    else:
        fig, ax = plt.subplots(figsize=figsize)

        model = models[0]

        df_model = df.reset_index().set_index("fml").xs(model)

        coef = df_model["Estimate"].values
        conf_l = coef - df_model["Std. Error"].values * norm.ppf(1 - alpha / 2)
        conf_u = coef + df_model["Std. Error"].values * norm.ppf(1 - alpha / 2)
        coefnames = df_model["Coefficient"].values.tolist()

        if is_iplot == True:
            fig.suptitle("iplot")
            coefnames = [
                (i)
                for string in coefnames
                for i in re.findall(r"\[T\.([\d\.\-]+)\]", string)
            ]

        ax.scatter(coefnames, coef, color="b", alpha=0.8)
        ax.scatter(coefnames, conf_u, color="b", alpha=0.8, marker="_", s=100)
        ax.scatter(coefnames, conf_l, color="b", alpha=0.8, marker="_", s=100)
        ax.vlines(coefnames, ymin=conf_l, ymax=conf_u, color="b", alpha=0.8)
        if yintercept is not None:
            ax.axhline(yintercept, color="red", linestyle="--", alpha=0.5)
        if xintercept is not None:
            ax.axvline(xintercept, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel("Coefficients")
        ax.set_title(model)
        ax.tick_params(axis="x", rotation=rotate_xticks)

        plt.show()
        plt.close()


def _check_ivars(data, ivars):
    """
    Checks if the variables in the i() syntax are of the correct type.
    Args:
        data (pandas.DataFrame): The dataframe containing the data used for the model fitting.
        ivars (list): The list of variables specified in the i() syntax.
    Returns:
        None
    """

    i0_type = data[ivars[0]].dtype
    i1_type = data[ivars[1]].dtype
    if not i0_type in ["category", "O"]:
        raise ValueError(
            "Column "
            + ivars[0]
            + " is not of type 'O' or 'category', which is required in the first position of i(). Instead it is of type "
            + i0_type.name
            + ". If a reference level is set, it is required that the variable in the first position of 'i()' is of type 'O' or 'category'."
        )
        if not i1_type in ["int64", "float64", "int32", "float32"]:
            raise ValueError(
                "Column "
                + ivars[1]
                + " is not of type 'int' or 'float', which is required in the second position of i(). Instead it is of type "
                + i1_type.name
                + ". If a reference level is set, iti is required that the variable in the second position of 'i()' is of type 'int' or 'float'."
            )


def _prepare_split_estimation(split, fsplit, data, fml_dict):
    """
    Cleans the input for the split estimation.
    Checks if the split variables are of the correct type.

    Args:
        split (str): The name of the variable used for the split estimation.
        fsplit (str): The name of the variable used for the fixed split estimation.
        data (pandas.DataFrame): The dataframe containing the data used for the model fitting.
        var_dict (dict): The dictionary containing the variables used in the model.
    Returns:
        splitvar (pandas.Series): The series containing the split variable.
        splitvar_name (str): The name of the split variable. Either equal to split or fsplit.
        estimate_split_model (bool): Whether to estimate the split model.
        estimate_full_model (bool): Whether to estimate the full model.
    """

    if split is not None:
        if fsplit is not None:
            raise ValueError(
                "Cannot specify both split and fsplit. Please specify only one of the two."
            )
        else:
            splitvar = data[split]
            estimate_full_model = False
            estimate_split_model = True
            splitvar_name = split
    elif fsplit is not None:
        splitvar = data[fsplit]
        splitvar_name = fsplit
        estimate_full_model = False
        estimate_split_model = True
    else:
        splitvar = None
        splitvar_name = None
        estimate_split_model = False
        estimate_full_model = True

    if splitvar is not None:
        split_categories = np.unique(splitvar)
        if splitvar_name not in data.columns:
            raise ValueError("Split variable " + splitvar + " not found in data.")
        if splitvar_name in fml_dict.keys():
            raise ValueError(
                "Split variable " + splitvar + " cannot be a fixed effect variable."
            )
        if splitvar.dtype.name != "category":
            splitvar = pd.Categorical(splitvar)

    return splitvar, splitvar_name, estimate_split_model, estimate_full_model


def get_fml(depvar, covar, fval, endogvars = None, instruments = None) -> str:

    '''
    Stiches together the formula string for the regression.

    Args:
        depvar (str): The dependent variable.
        covar (str): The covariates. E.g. "X1+X2+X3"
        fval (str): The fixed effects. E.g. "X1+X2". "0" if no fixed effects.
        endogvars (str): The endogenous variables.
        instruments (str): The instruments. E.g. "Z1+Z2+Z3"
    Returns:
        fml (str): The formula string for the regression.
    '''

    fml = depvar + " ~ " + covar

    if endogvars is not None:
        fml_iv = "|" + endogvars + "~" + instruments
    else:
        fml_iv = None

    if fval != "0":
        fml_fval = "|" + fval
    else:
        fml_fval = None

    if fml_fval is not None:
        fml += fml_fval

    if fml_iv is not None:
        fml += fml_iv


    fml = fml.replace(" ","")


    return fml



    if fval != "0":
        fml = depvar + " ~ " + covar + " | " + fval
    else:
        fml = depvar + " ~ " + covar

    return fml.replace(" ", "")


def _multicollinearity_checks(X, Z, ivars, fml2):
    """
    Checks for multicollinearity in the design matrices X and Z.
    Args:
        X (numpy.ndarray): The design matrix X.
        Z (numpy.ndarray): The design matrix (with instruments) Z.
        ivars (list): The list of variables specified in the i() syntax.
        fml2 (str): The formula string.

    """

    if np.linalg.matrix_rank(X) < min(X.shape):
        if ivars is not None:
            raise MatrixNotFullRankError(
                'The design Matrix X does not have full rank for the regression with fml" + fml2 + "."'
                "The model is skipped."
                "As you are running a regression via `i()` syntax, maybe you need to drop a level via i(var1, var2, ref = ...)?"
            )
        else:
            raise MatrixNotFullRankError(
                'The design Matrix X does not have full rank for the regression with fml" + fml2 + "."'
                "The model is skipped. "
            )

    if np.linalg.matrix_rank(Z) < min(Z.shape):
        if ivars is not None:
            raise MatrixNotFullRankError(
                'The design Matrix Z does not have full rank for the regression with fml" + fml2 + "."'
                "The model is skipped."
                'As you are running a regression via `i()` syntax, maybe you need to drop a level via i(var1, var2, ref = ...)?"'
            )
        else:
            raise MatrixNotFullRankError(
                'The design Matrix Z does not have full rank for the regression with fml" + fml2 + "."'
                "The model is skipped."
            )


def _get_vcov_type(vcov, fval):
    """
    Passes the specified vcov type. If no vcov type specified, sets the default vcov type as iid if no fixed effect
    is included in the model, and CRV1 clustered by the first fixed effect if a fixed effect is included in the model.
    Args:
        vcov (str): The specified vcov type.
        fval (str): The specified fixed effects. (i.e. "X1+X2")
    Returns:
        vcov_type (str): The specified vcov type.
    """

    if vcov is None:
        # iid if no fixed effects
        if fval == "0":
            vcov_type = "iid"
        else:
            # CRV1 inference, clustered by first fixed effect
            first_fe = fval.split("+")[0]
            vcov_type = {"CRV1": first_fe}
    else:
        vcov_type = vcov

    return vcov_type


def _clean_ivars(ivars, data):
    """
    Clean variables interacted via i(X1, X2, ref = a) syntax.

    Args:
        ivars (list): The list of variables specified in the i() syntax.
        data (pandas.DataFrame): The dataframe containing the data used for the model fitting.
    Returns:
        ivars (list): The list of variables specified in the i() syntax minus the reference level
        drop_ref (str): The dropped reference level specified in the i() syntax. None if no level is dropped
    """

    if ivars is not None:
        if list(ivars.keys())[0] is not None:
            ref = list(ivars.keys())[0]
            ivars = ivars[ref]
            drop_ref = ivars[0] + "[T." + ref + "]" + ":" + ivars[1]
        else:
            ivars = ivars[None]
            drop_ref = None

        # type checking for ivars variable
        _check_ivars(data, ivars)

    else:
        ivars = None
        drop_ref = None

    return ivars, drop_ref


def _drop_singletons(fixef_rm):
    """
    Checks if the fixef_rm argument is set to "singleton". If so, returns True, else False.
    Args:
        fixef_rm (str): The fixef_rm argument.
    Returns:
        drop_singletons (bool): Whether to drop singletons.
    """

    if fixef_rm == "singleton":
        return True
    else:
        return False


def _find_untransformed_depvar(transformed_depvar):
    """
    Args:
        transformed_depvar (str): The transformed depvar

    find untransformed depvar in a formula
    i.e. if "a" is transormed to "log(a)", then "a" is returned
    """

    match = re.search(r"\((.*?)\)", transformed_depvar)
    if match:
        return match.group(1)
    else:
        return transformed_depvar
