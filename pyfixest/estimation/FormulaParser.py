import re
from itertools import product
from typing import Optional, Union

from pyfixest.errors import (
    DuplicateKeyError,
    EndogVarsAsCovarsError,
    InstrumentsAsCovarsError,
    UnderDeterminedIVError,
    UnsupportedMultipleEstimationSyntax,
)


class FixestFormulaParser:
    """
    A class for deparsing formulas with multiple estimation syntax.

    Methods
    -------
        __init__(self, fml): Constructor method that initializes the object with
        a given formula.
        get_fml_dict(self): Returns a dictionary of all fevars & formula without
        fevars.
        get_var_dict(self): Returns a dictionary of all fevars and list of covars
        and depvars used in regression with those fevars.

    """

    def __init__(self, fml: str):
        """
        Initialize the object with a given formula using constructor method.

        Parameters
        ----------
        fml : str
            A one-to three sided formula string in the form
            "Y1 + Y2 ~ X1 + X2 | FE1 + FE2 | endogvar ~ exogvar".

        Returns
        -------
            None

        """
        depvars, covars, fevars, endogvars, instruments = _deparse_fml(fml)

        # Parse all individual formula components that allow for
        # multiple estimations into dictionaries that separate a 'constant'
        # part common to all estimations from a varying part with key of the
        # 'type' of variation, e.g. 'sw' or 'csw'.
        depvars_list = re.split(r"\s*\+\s*", depvars)
        covars_dict = _input_formula_to_dict(
            covars
        )  # e.g. {'constant': [], 'csw': ['X1', 'X2']}
        fevars_dict = _input_formula_to_dict(fevars)  # e.g. {'constant': ['f1^f2']}
        # Now parse all formula components in covars_dict, fevars_dict into lists
        # of formulas that can be used in the estimation.
        # E.g. {'constant': [], 'csw': ['X1', 'X2']} becomes ['X1', 'X1+X2']
        # and {'constant': ['f1^f2']} becomes ['f1^f2'].
        covars_formulas_list = _dict_to_list_of_formulas(
            covars_dict
        )  # evaluate self.covars to list: ['X1', 'X1+X2']
        fevars_formula_list = _dict_to_list_of_formulas(fevars_dict)  # ['f1^f2']

        self.condensed_fml_dict = collect_fml_dict(
            fevars_formula_list, depvars_list, covars_formulas_list
        )

        # now repeat for IV:
        self.is_iv = False
        self.condensed_fml_dict_iv = None
        endogvars_list = []
        instruments_formulas_list = []
        if endogvars is not None and instruments is not None:
            self.is_iv = True
            endogvars_list = re.split(r"\s*\+\s*", endogvars)
            instruments_dict = _input_formula_to_dict(instruments)
            instruments_formulas_list = _dict_to_list_of_formulas(instruments_dict)
            self.condensed_fml_dict_iv = collect_fml_dict(
                fevars_formula_list, endogvars_list, instruments_formulas_list
            )

        self.FixestFormulaDict: dict[str, list[FixestFormula]] = {}
        self.populate_fixest_formula_dict(
            depvars_list=depvars_list,
            covars_formulas_list=covars_formulas_list,
            fevars_formula_list=fevars_formula_list,
            endogvars_list=endogvars_list,
            instruments_formulas_list=instruments_formulas_list,
        )

    def add_to_FixestFormulaDict(
        self,
        depvar: str,
        covar: str,
        fval: str,
        endogvar: Optional[str] = None,
        instrument: Optional[str] = None,
    ):
        """
        Add a FixestFormula object to the FixestFormulaDict.

        This method initializes a FixestFormula object with the specified
        dependent variable, covariates, fixed effect variable, optional endogenous
        variable, and optional instrument. It then generates both first and
        second stage formulas, as well as the combined formula,
        and adds the FixestFormula object to the FixestFormulaDict
        under the key specified by the `fval` parameter.
        If the `fval` key does not exist in the dictionary, it is created.

        Parameters
        ----------
        depvar : str
            The name of the dependent variable.
        covar : str
            The covariates included in the model, formatted as a string. E.g. X1+X2.
        fval : str
            The fixed effect variable included in the model.
        endogvar : Optional[str], optional
            The endogenous variable, if applicable.
        instrument : Optional[str], optional
            The instrument for the endogenous variable, if applicable.

        Notes
        -----
        - This method does not return any value.
        - It updates the `FixestFormulaDict` attribute of the instance by
          appending a new FixestFormula object or creating a new list for
          a previously unused `fval`.
        """
        FixestFML = FixestFormula(
            depvar=depvar,
            covar=covar,
            fval=fval,
            endogvars=endogvar,
            instruments=instrument,
        )
        FixestFML.get_first_and_second_stage_fml()
        FixestFML.get_fml()

        if fval not in self.FixestFormulaDict:
            self.FixestFormulaDict[fval] = []
        self.FixestFormulaDict[fval].append(FixestFML)

    def populate_fixest_formula_dict(
        self,
        depvars_list: list[str],
        covars_formulas_list: list[str],
        fevars_formula_list: list[str],
        endogvars_list: list[str],
        instruments_formulas_list: list[str],
    ):
        """
        Populate the FixestFormulaDict with FixestFormula objects.

        Iterates through all combinations of provided lists of dependent
        variables, covariate formulas, fixed effect variables, endogenous
        variables (if applicable), and instrument formulas (if applicable)
        to populate the `FixestFormulaDict` with corresponding FixestFormula
        objects. This method supports models with and without instrumental
        variables (IV).

        Parameters
        ----------
        depvars_list : list
            A list of dependent variable names.
        covars_formulas_list : list
            A list of covariate formulas.
        fevars_formula_list : list
            A list of fixed effect variable formulas.
        endogvars_list : list
            A list of endogenous variables, used if the model is an IV model.
        instruments_formulas_list : list
            A list of instrument formulas, used if the model is an IV model.

        Notes
        -----
        - This method does not return any value.
        - It conditionally processes combinations based on whether the model is
          identified as an IV model, indicated by the `is_iv` attribute of the
          instance.
        - For IV models, it combines variables from `depvars_list`,
          `covars_formulas_list`, `fevars_formula_list`,
          `endogvars_list`, and `instruments_formulas_list`.
        - For non-IV models, it combines variables from `depvars_list`,
          `covars_formulas_list`, and `fevars_formula_list` only.
        """
        if self.is_iv:
            for depvar, covar, fval, endogvar, instrument in product(
                depvars_list,
                covars_formulas_list,
                fevars_formula_list,
                endogvars_list,
                instruments_formulas_list,
            ):
                self.add_to_FixestFormulaDict(depvar, covar, fval, endogvar, instrument)
        else:
            for depvar, covar, fval in product(
                depvars_list, covars_formulas_list, fevars_formula_list
            ):
                self.add_to_FixestFormulaDict(depvar, covar, fval)

    def set_fixest_multi_flag(self):
        """
        Set a flag to indicate whether multiple estimations are being performed or not.

        Simple check if `all_fitted_models` has length greater than 1.
        Throws an error if multiple estimations are being performed with IV estimation.
        Args:
            None
        Returns:
            None
        """
        if (
            len(self.FixestFormulaDict) == 1
            and isinstance(next(iter(self.FixestFormulaDict.values())), list)
            and len(next(iter(self.FixestFormulaDict.values()))) == 1
        ):
            self._is_multiple_estimation = False
        else:
            self._is_multiple_estimation = True
            if self.is_iv:
                raise NotImplementedError(
                    """
                    Multiple Estimations is currently not supported with IV.
                    This is mostly due to insufficient testing and will be possible
                    with a future release of PyFixest.
                    """
                )


class FixestFormula:
    """
    A class with information contained in model formulas.

    Attributes
    ----------
    _depvar : str
        The dependent variable in the model.
    _covar : str
        The covariates in the model, separated by '+'.
    _fval : str
        An optional fixed effect variable included in the model.
        Separated by "+". "0" if no fixed effect in the model.
    _endogvars : str, optional
        Endogenous variables in the model, separated by '+'.
    _instruments : str, optional
        Instrumental variables for the endogenous variables, separated by '+'.

    Methods
    -------
    get_first_and_second_stage_fml(self):
        Assigns first and second stage formulas for IV models
        to instance variables.

    get_fml(self):
        Constructs and stores a formula based on instance attributes.

    check_syntax(self):
        Validates the formula by checking if instruments are incorrectly
        specified as covariates.

    Raises
    ------
    InstrumentsAsCovarsError
        If any instrumental variables are also specified as covariates
        in the model formula.

    """

    def __init__(
        self,
        depvar: str,
        covar: str,
        fval: Optional[str] = None,
        endogvars: Optional[str] = None,
        instruments: Optional[str] = None,
    ):
        self._depvar = depvar
        self._covar = covar
        if fval is None:
            fval = "0"
        self._fval = fval
        self._endogvars = endogvars
        self._instruments = instruments

    def get_first_and_second_stage_fml(self):
        """Assign first and second stage formulas."""
        self.fml_second_stage, self.fml_first_stage = _get_first_and_second_stage_fml(
            depvar=self._depvar,
            covar=self._covar,
            fval=self._fval,
            endogvar=self._endogvars,
            instruments=self._instruments,
        )

    def get_fml(self):
        """
        Construct and stores a Wilkinson formula..

        This method combines dependent variable, covariates, endogenous variables,
        instrumental variables, and an optional fixed value to construct a statistical
        model formula. This formula is then stored in the instance's `fml` attribute.
        The general structure of the formula is
        `depvar ~ covar | fval | endogvars ~ instruments`.
        Spaces in the formula are removed before storing.

        Attributes Used
        ---------------
        _depvar : str
            The dependent variable in the model.
        _covar : str
            The covariates in the model, separated by '+'.
        _fval : str
            An optional fixed value to be included in the model.
            If set to "0", it is ignored.
        _endogvars : str, optional
            Endogenous variables in the model, separated by '+'.
            If `None`, this part of the formula is omitted.
        _instruments : str, optional
            Instrumental variables in the model, separated by '+'.
            Relevant only if `_endogvars` is not `None`.

        Notes
        -----
        - The method does not return any value but updates the instance's `fml`
          attribute with the constructed formula.
        - This method assumes that `_depvar`, `_covar`, and `_fval` are always
          provided and treats `_endogvars` and `_instruments` as optional components
          of the formula.
        """
        depvar = self._depvar
        covar = self._covar
        fval = self._fval
        endogvars = self._endogvars
        instruments = self._instruments

        fml = f"{depvar} ~ {covar}"
        fml_iv = f"| {endogvars} ~ {instruments}" if endogvars is not None else None

        fml_fval = f"| {fval}" if fval != "0" else None

        if fml_iv is not None:
            fml += fml_iv

        if fml_fval is not None:
            fml += fml_fval

        self.fml = fml.replace(" ", "")

    def check_syntax(self):
        """
        Check if any instrument variables are mistakenly specified as covariates.

        This method processes the instrument and covariate strings stored in
        the instance's `_instruments` and `_covar` attributes, respectively.
        It splits these strings by '+' to identify individual variables. If any
        variable is found to be listed as both an instrument and a covariate,
        an `InstrumentsAsCovarsError` is raised, indicating that
        instrument variables cannot be specified as covariates.

        Raises
        ------
        InstrumentsAsCovarsError
            If any instrument variables are also specified as covariates.
            The error message includes the names of the variables causing
            this conflict.

        Notes
        -----
        - The `_instruments` and `_covar` attributes must be strings containing variable
          names separated by '+'. If either attribute is `None`, this check is skipped.
        - This method does not return any value but raises an error if the check fails.
        """
        instruments = self._instruments
        covars = self._covar

        if instruments is not None:
            instruments_as_covars = [
                element
                for element in re.split(r"\s*\+\s*", instruments)
                if element in re.split(r"\s*\+\s*", covars)
            ]

            if instruments_as_covars:
                raise InstrumentsAsCovarsError(
                    f"""
                    The instrument(s) {",".join(instruments_as_covars)} are specified as
                    covariates in the first part of the three-part formula. This is not allowed.
                    """
                )


def _get_first_and_second_stage_fml(
    depvar: str,
    covar: str,
    fval: str,
    endogvar: Optional[str] = None,
    instruments: Optional[str] = None,
) -> tuple:
    """
    Generate first and second stage formulas for OLS and IV regression models.

    Parameters
    ----------
    depvar : str
        Dependent variable.
    covar : str
        Covariates.
    fval : str, optional
        Fixed effect variable, defaults to "0" if None.
    endogvar : str, optional
        Endogenous variable.
    instruments : str
        Instrumental variables.

    Returns
    -------
    tuple
        (Second stage formula, First stage formula).

    Notes
    -----
    First stage formula is None if `endogvar` is not provided.
    """
    if fval is None:
        fval = "0"

    fml_iv = f"{endogvar} ~ {instruments}"

    fml_second_stage = f"{depvar} ~ {covar} + 1"
    fml_first_stage = f"{fml_iv}+{covar}-{endogvar} + 1" if endogvar else None

    return fml_second_stage, fml_first_stage


def collect_fml_dict(
    fevars_formula: list[str], depvars_dict: list[str], covars_formula: list[str]
):
    """
    Create a nested dictionary mapping fixed effects to depvar-covariate formulas.

    Parameters
    ----------
    fevars_formula : list
        Fixed effects variables formulas.
    depvars_dict : dict
        Mapping of dependent variables to their properties.
    covars_formula : list
        Covariate formulas.

    Returns
    -------
    dict
        Nested dict of fevar to depvar to list of covariate formulas.
    """
    fml_dict: dict[str, dict[str, list[str]]] = {}

    for fevar in fevars_formula:
        res: dict[str, list[str]] = {}
        for depvar in depvars_dict:
            res[depvar] = []
            for covar in covars_formula:
                res[depvar].append(f"{depvar}~{covar}")
        fml_dict[fevar] = res

    return fml_dict


def _deparse_fml(
    fml: str,
) -> tuple[str, str, str, Union[str, None], Union[str, None]]:
    """
    Decompose a formula string into its constituent parts.

    This function takes a formula string and splits it into its components based on
    the presence of '~' and '|' characters. The formula string is expected to follow
    the format 'depvars ~ covars | fevars | endogvars ~ instruments'.

    Parameters
    ----------
    fml : str
        The formula string to be decomposed.

    Returns
    -------
    tuple
        A tuple containing the decomposed parts of the formula as strings:
        (depvars, covars, fevars, endogvars, instruments).
        `endogvars` and `instruments` may be `None` if not applicable.

    Raises
    ------
    UnderDeterminedIVError
        If the number of instruments is less than the number of endogenous variables,
        indicating an underdetermined IV system.

    Notes
    -----
    - Fixed effects variables are set to "0" if not explicitly provided in
      the formula.
    - The function automatically adds endogenous variables to the covariates
      list, a behavior indicated by the comment on potentially misleading
      naming conventions.

    Examples
    --------
    ```{python}
    fml = "y ~ x1+x2|z1+z2|w1 ~ w2+w3"
    _deparse_fml(fml)
    ('y', 'w1+x1+x2', 'z1+z2', 'w1', 'w2+w3')
    ```

    Here, `y` is the dependent variable, `x1+x2` are the covariates
    (with `w1` added due to it being an endogenous variable), `z1+z2` are
    the fixed effects variables, and `w1` is the endogenous
    variable with `w2+w3` as its instruments.
    """
    # Split the formula string into its components
    fml_split = re.split(r"\s*\|\s*", fml.strip())
    depvars, covars = re.split(r"\s*~\s*", fml_split[0])

    if len(fml_split) == 1:
        fevars = "0"
        endogvars = None
        instruments = None
    elif len(fml_split) == 2:
        if "~" in fml_split[1]:
            fevars = "0"
            endogvars, instruments = re.split(r"\s*~\s*", fml_split[1])
            # add endogenous variable to "covars" - yes, bad naming
            _check_endogvars_as_covars(endogvars, covars)
            covars = endogvars if covars == "1" else f"{endogvars}+{covars}"
        else:
            fevars = fml_split[1]
            endogvars = None
            instruments = None
    elif len(fml_split) == 3:
        fevars = fml_split[1]
        endogvars, instruments = re.split(r"\s*~\s*", fml_split[2])
        _check_endogvars_as_covars(endogvars, covars)

        # add endogenous variable to "covars" - yes, bad naming
        covars = endogvars if covars == "1" else f"{endogvars}+{covars}"

    endogvars_list = []
    instruments_list = []

    if endogvars is not None and not isinstance(endogvars, list):
        endogvars_list = re.split(r"\s*\+\s*", endogvars)

    if instruments is not None and not isinstance(instruments, list):
        instruments_list = re.split(r"\s*\+\s*", instruments)

    if endogvars_list and instruments_list:
        if len(endogvars_list) > len(instruments_list):
            raise UnderDeterminedIVError(
                """
                The IV system is underdetermined. Please provide as many or
                more instruments as endogenous variables.
                """
            )
        else:
            pass

    return depvars, covars, fevars, endogvars, instruments


def _check_endogvars_as_covars(endogvars: str, covars: str):
    """
    Check if one or more endogenous variables are included in the covariates.

    Parameters
    ----------
    endogvars : str
        A string representing the endogenous variables in the model,
        separated by "+".
    covars : str
        A string representing the covariates in the model, separated by "+".

    Raises
    ------
    EndogVarsAsCovarsError
        If any of the specified endogenous variables are also listed as covariates.

    Returns
    -------
    None
    """
    endogvars_as_covars = [
        element
        for element in re.split(r"\s*\+\s*", endogvars)
        if element in re.split(r"\s*\+\s*", covars)
    ]

    if endogvars_as_covars:
        raise EndogVarsAsCovarsError(
            f"""
                The endogeneous variable(s) {",".join(endogvars_as_covars)} are specified as
                covariates in the first part of the three-part formula. This is not allowed.
                """
        )


def _input_formula_to_dict(x: str) -> dict[str, list[str]]:
    """
    Parse a formula string.

    Given a formula string `x` - e.g. 'X1 + csw(X2, X3)' - , splits it into its
    constituent variables and their types (if any), and returns a dictionary
    containing the result. The dictionary has the following keys: 'constant',
    'sw', 'sw0', 'csw'.

    The values are lists of variables of the respective type.

    Parameters
    ----------
    x : str
        The formula string to unpack.

    Returns
    -------
    res_s : dict
        A dictionary containing the unpacked formula. The dictionary has the
        following keys:
            - 'constant' : list of str
                The list of constant (i.e., non-switched) variables in the formula.
            - 'sw' : list of str
                The list of variables that have a regular switch
                (i.e., 'sw(var1, var2, ...)' notation) in the formula.
            - 'sw0' : list of str
                The list of variables that have a 'sw0(var1, var2, ..)' switch
                in the formula.
            - 'csw' : list of str or list of lists of str
                The list of variables that have a 'csw(var1, var2, ..)' switch
                in the formula.
                Each element in the list can be either a single variable string,
                or a list of variable strings if multiple variables are listed
                in the switch.
            - 'csw0' : list of str or list of lists of str
                The list of variables that have a 'csw0(var1,var2,...)' switch
                in the formula.
                Each element in the list can be either a single variable string,
                or a list of variable strings if multiple variables are listed
                in the switch.

    Raises
    ------
    ValueError:
        If the switch type is not one of 'sw', 'sw0', 'csw', or 'csw0'.

    Example:
    --------
    >>> _input_formula_to_dict("a+sw(b)+csw(x1,x2)+sw0(d)+csw0(y1,y2,y3)")
    {'constant': ['a'],
     'sw': ['b'],
     'csw': [['x1', 'x2']],
     'sw0': ['d'],
     'csw0': [['y1', 'y2', 'y3']]}
    """
    # Split the formula into its constituent variables
    var_split = re.split(r"\s*\+\s*", x)

    res_s: dict[str, list[str]] = {"constant": []}
    for var in var_split:
        # Check if this variable contains a switch
        varlist, sw_type = _find_multiple_estimation_syntax(var)

        # If there's no switch, just add the variable to the list
        if sw_type is None:
            res_s["constant"].append(varlist)

        # If there'_ a switch, unpack it and add it to the list
        else:
            if sw_type in ["sw", "sw0", "csw", "csw0"]:
                _check_duplicate_key(res_s, sw_type)
                res_s[sw_type] = varlist
            elif sw_type == "varying_slopes":
                res_s[sw_type] = varlist
            else:
                raise UnsupportedMultipleEstimationSyntax("Unsupported switch type")

    # Sort the list by type (strings first, then lists)
    # res_s.sort(key=lambda x: 0 if isinstance(x, str) else 1)

    return res_s


def _dict_to_list_of_formulas(unpacked: dict[str, list[str]]) -> list[str]:
    """
    Generate a list of formula strings from a dictionary of "unpacked" fml vars.

    Given a dictionary of "unpacked" formula variables, returns a string containing
    formulas. An "unpacked" formula is a deparsed formula that allows for multiple
    estimations.

    Parameters
    ----------
    unpacked : dict
        A dictionary of unpacked formula variables. The dictionary has the following
        keys:
            - 'constant' : list of str
                The list of constant (i.e., non-switched) variables in the formula.
            - 'sw' : list of str
                The list of variables that have a regular switch
                (i.e., 'sw(var1, var2, ...)' notation) in the formula.
            - 'sw0' : list of str
                The list of variables that have a 'sw0(var1, var2, ..)' switch
                in the formula.
            - 'csw' : list of str or list of lists of str
                The list of variables that have a 'csw(var1, var2, ..)' switch
                in the formula.
                Each element in the list can be either a single variable string,
                or a list of variable strings if multiple variables are listed
                in the switch.
            - 'csw0' : list of str or list of lists of str
    """
    res = {"constant": unpacked.get("constant", [])}

    # add up all variable constants (only required for csw)
    if "csw" in unpacked:
        res["variable"] = unpacked["csw"]
        variable_type = "csw"
    elif "csw0" in unpacked:
        res["variable"] = unpacked["csw0"]
        variable_type = "csw0"
    elif "sw" in unpacked:
        res["variable"] = unpacked["sw"]
        variable_type = "sw"
    elif "sw0" in unpacked:
        res["variable"] = unpacked["sw0"]
        variable_type = "sw0"
    else:
        res["variable"] = []
        variable_type = None

    const_fml = "+".join(res["constant"]) if res["constant"] else ""

    variable_fml = []
    if res["variable"]:
        if variable_type in ["csw", "csw0"]:
            variable_fml = [
                "+".join(res["variable"][: i + 1]) for i in range(len(res["variable"]))
            ]
        else:
            variable_fml = [res["variable"][i] for i in range(len(res["variable"]))]
        if variable_type in ["sw0", "csw0"]:
            variable_fml = ["0", *variable_fml]

    fml_list = []
    if variable_fml:
        if const_fml:
            fml_list = [
                f"{const_fml}+{variable_fml[i]}"
                for i in range(len(variable_fml))
                if variable_fml[i] != "0"
            ]
            if variable_type in ["sw0", "csw0"]:
                fml_list.insert(0, const_fml)
        else:
            fml_list = variable_fml
    elif const_fml:
        fml_list.append(const_fml)
    else:
        raise AttributeError("Not a valid formula provided.")

    if not isinstance(fml_list, list):
        fml_list = [fml_list]

    return fml_list


def _find_multiple_estimation_syntax(x: str):
    """
    Search for matches of multiple estimation syntax in a string.

    Matches are either 'sw', 'sw0', 'csw', 'csw0'. If a match is found, returns a
    tuple containing a list of the elements found and the type of match. Otherwise,
    returns the original string and None.

    Parameters
    ----------
    x : str
        The string to search for matches in.

    Returns
    -------
    list[str] or str, str or None
        If any matches were found, returns a
        tuple containing a list of the elements found and the type of match
        (either 'sw', 'sw0', 'csw', or 'csw0').
        Otherwise, returns the original string and None.

    Example:
        _find_multiple_estimation_syntax('sw(var1, var2)') -> (['var1', ' var2'], 'sw')
    """
    # Search for matches in the string
    sw_match = re.findall(r"sw\((.*?)\)", x)
    csw_match = re.findall(r"csw\((.*?)\)", x)
    sw0_match = re.findall(r"sw0\((.*?)\)", x)
    csw0_match = re.findall(r"csw0\((.*?)\)", x)

    # Check for sw matches
    if sw_match:
        if csw_match:
            return csw_match[0].split(","), "csw"
        else:
            return sw_match[0].split(","), "sw"

    # Check for sw0 matches
    elif sw0_match:
        if csw0_match:
            return csw0_match[0].split(","), "csw0"
        else:
            return sw0_match[0].split(","), "sw0"

    # No matches found
    else:
        return x, None


def _check_duplicate_key(my_dict: dict, key: str):
    """
    Identify duplicate keys.

    Checks if a key already exists in a dictionary. If it does, raises a
    DuplicateKeyError. Otherwise, does nothing.

    Parameters
    ----------
    my_dict : dict
        The dictionary to check for duplicate keys.
    key : str
        The key to check for in the dictionary.

    Returns
    -------
        None
    """
    for key in ["sw", "csw", "sw0", "csw0"]:
        if key in my_dict:
            raise DuplicateKeyError(
                f"""
                Duplicate key found: "{key}. Multiple estimation syntax can
                only be used once on the rhs of the two-sided formula.
                """
            )
