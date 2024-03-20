import re
from itertools import product
from typing import Optional

from pyfixest.errors import (
    DuplicateKeyError,
    UnderDeterminedIVError,
    UnsupportedMultipleEstimationSyntax,
    InstrumentsAsCovarsError,
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

    def __init__(self, fml):
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
        depvars, covars, fevars, endogvars, instruments = deparse_fml(fml)

        # Parse all individual formula components that allow for
        # multiple estimations into dictionaries that separate a 'constant'
        # part common to all estimations from a varying part with key of the
        # 'type' of variation, e.g. 'sw' or 'csw'.
        depvars_list = depvars.split("+")
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
        if endogvars is not None:
            self.is_iv = True
            endogvars_list = endogvars.split("+")
            instruments_dict = _input_formula_to_dict(instruments)
            instruments_formulas_list = _dict_to_list_of_formulas(instruments_dict)
            self.condensed_fml_dict_iv = collect_fml_dict(
                fevars_formula_list, endogvars_list, instruments_formulas_list
            )

        self.FixestFormulaDict = {}
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
        depvars_list,
        covars_formulas_list,
        fevars_formula_list,
        endogvars_list,
        instruments_formulas_list,
    ):

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


class FixestFormula:

    def __init__(self, depvar, covar, fval, endogvars, instruments):

        self._depvar = depvar
        self._covar = covar
        self._fval = fval
        self._endogvars = endogvars
        self._instruments = instruments

    def get_first_and_second_stage_fml(self):

        self.fml_second_stage, self.fml_first_stage = _get_first_and_second_stage_fml(
            depvar=self._depvar,
            covar=self._covar,
            fval=self._fval,
            endogvar=self._endogvars,
            instruments=self._instruments,
        )

    def get_fml(self):

        depvar = self._depvar
        covar = self._covar
        fval = self._fval
        endogvars = self._endogvars
        instruments = self._instruments

        fml = f"{depvar} ~ {covar}"
        fml_iv = f"| {endogvars} ~ {instruments}" if endogvars is not None else None

        fml_fval = f"| {fval}" if fval != "0" else None

        if fml_fval is not None:
            fml += fml_fval

        if fml_iv is not None:
            fml += fml_iv

        self.fml = fml.replace(" ", "")

    def check_syntax(self):

        instruments = self._instruments
        covars = self._covar

        if instruments is not None:

            instruments_as_covars = [
                element
                for element in instruments.split("+")
                if element in covars.split("+")
            ]

            if instruments_as_covars:
                raise InstrumentsAsCovarsError(
                    f"""
                    The instrument(s) {",".join(instruments_as_covars)} are specified as
                    covariates in the first part of the three-part formula. This is not allowed.
                    """
                )


def _get_first_and_second_stage_fml(
    depvar,
    covar,
    fval,
    endogvar,
    instruments,
):

    if fval is None:
        fval = "0"

    fml_iv = f"{endogvar} ~ {instruments}"

    fml_second_stage = f"{depvar} ~ {covar} + 1"
    fml_first_stage = f"{fml_iv}+{covar}-{endogvar} + 1" if endogvar else None

    return fml_second_stage, fml_first_stage


def get_fml(
    depvar: str, covar: str, fval: str, endogvars: str = None, instruments: str = None
) -> str:
    """
    Stitches together the formula string for the regression.

    Parameters
    ----------
    depvar : str
        The dependent variable.
    covar : str
        The covariates. E.g. "X1+X2+X3"
    fval : str
        The fixed effects. E.g. "X1+X2". "0" if no fixed effects.
    endogvars : str, optional
        The endogenous variables.
    instruments : str, optional
        The instruments. E.g. "Z1+Z2+Z3"

    Returns
    -------
    str
        The formula string for the regression.
    """
    fml = f"{depvar} ~ {covar}"
    fml_iv = f"| {endogvars} ~ {instruments}" if endogvars is not None else None

    fml_fval = f"| {fval}" if fval != "0" else None

    if fml_fval is not None:
        fml += fml_fval

    if fml_iv is not None:
        fml += fml_iv

    fml = fml.replace(" ", "")

    return fml


def collect_fml_dict(fevars_formula, depvars_dict, covars_formula):
    """
    Condense the formulas into a nested dictionary.

    Parameters
    ----------
    """
    fml_dict = {}

    for fevar in fevars_formula:
        res = {}
        for depvar in depvars_dict:
            res[depvar] = []
            for covar in covars_formula:
                res[depvar].append(f"{depvar}~{covar}")
        fml_dict[fevar] = res

    return fml_dict


def deparse_fml(fml):

    # Clean up the formula string
    fml = "".join(fml.split())

    # Split the formula string into its components
    fml_split = fml.split("|")
    depvars, covars = fml_split[0].split("~")

    if len(fml_split) == 1:
        fevars = "0"
        endogvars = None
        instruments = None
    elif len(fml_split) == 2:
        if "~" in fml_split[1]:
            fevars = "0"
            endogvars, instruments = fml_split[1].split("~")
            # add endogenous variable to "covars" - yes, bad naming

            covars = endogvars if covars == "1" else f"{endogvars}+{covars}"
        else:
            fevars = fml_split[1]
            endogvars = None
            instruments = None
    elif len(fml_split) == 3:
        fevars = fml_split[1]
        endogvars, instruments = fml_split[2].split("~")

        # add endogenous variable to "covars" - yes, bad naming
        covars = endogvars if covars == "1" else f"{endogvars}+{covars}"

    if endogvars is not None:
        if not isinstance(endogvars, list):
            endogvars_list = endogvars.split("+")
        if not isinstance(instruments, list):
            instruments_list = instruments.split("+")
        if len(endogvars_list) > len(instruments_list):
            raise UnderDeterminedIVError(
                "The IV system is underdetermined. Please provide as many or more instruments as endogenous variables."
            )
        else:
            pass

    return depvars, covars, fevars, endogvars, instruments


def _input_formula_to_dict(x):
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
    var_split = x.split("+")

    res_s = {"constant": []}
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


def _dict_to_list_of_formulas(unpacked):
    """
    Generate a list of formula strings from a dictionary of "unpacked" formula variables.

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

    const_fml = "+".join(res["constant"]) if res["constant"] else []

    variable_fml = []
    if res["variable"]:
        if variable_type in ["csw", "csw0"]:
            variable_fml = [
                "+".join(res["variable"][: i + 1]) for i in range(len(res["variable"]))
            ]
        else:
            variable_fml = [res["variable"][i] for i in range(len(res["variable"]))]
        if variable_type in ["sw0", "csw0"]:
            variable_fml = ["0"] + variable_fml

    fml_list = []
    if variable_fml:
        if const_fml:
            fml_list = [
                f"{const_fml}+{variable_fml[i]}"
                for i in range(len(variable_fml))
                if variable_fml[i] != "0"
            ]
            if variable_type in ["sw0", "csw0"]:
                fml_list = [const_fml] + fml_list
        else:
            fml_list = variable_fml
    elif const_fml:
        fml_list = const_fml
    else:
        raise AttributeError("Not a valid formula provided.")

    if not isinstance(fml_list, list):
        fml_list = [fml_list]

    return fml_list


def _find_multiple_estimation_syntax(x):
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


def _check_duplicate_key(my_dict, key):
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
        else:
            None
