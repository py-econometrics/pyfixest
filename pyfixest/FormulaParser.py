import re
from pyfixest.exceptions import (
    DuplicateKeyError,
    EndogVarsAsCovarsError,
    InstrumentsAsCovarsError,
    UnderDeterminedIVError,
    UnsupportedMultipleEstimationSyntax,
)


class FixestFormulaParser:

    """
    A class for parsing a formula string into its individual components.

    Attributes:
        depvars (list): A list of dependent variables in the formula.
        covars (list): A list of covariates in the formula.
        fevars (list): A list of fixed effect variables in the formula.
        covars_fml (str): A string representation of the covariates in the formula.
        fevars_fml (str): A string representation of the fixed effect variables in the formula.

    Methods:
        __init__(self, fml): Constructor method that initializes the object with a given formula.
        get_fml_dict(self): Returns a dictionary of all fevars & formula without fevars.
        get_var_dict(self): Returns a dictionary of all fevars and list of covars and depvars used in regression with those fevars.

    """

    def __init__(self, fml):
        """
        Constructor method that initializes the object with a given formula.

        Args:
        fml (str): A two-formula string in the form "Y1 + Y2 ~ X1 + X2 | FE1 + FE2".

        Returns:
            None

        """

        # fml =' Y + Y2 ~  i(X1, X2) |csw0(X3, X4)'

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
                # add endogeneous variable to "covars" - yes, bad naming

                # check if any of the instruments or endogeneous variables are also specified
                # as covariates
                if any(
                    element in covars.split("+") for element in endogvars.split("+")
                ):
                    raise EndogVarsAsCovarsError(
                        "Endogeneous variables are specified as covariates in the first part of the three-part formula. This is not allowed."
                    )

                if any(
                    element in covars.split("+") for element in instruments.split("+")
                ):
                    raise InstrumentsAsCovarsError(
                        "Instruments are specified as covariates in the first part of the three-part formula. This is not allowed."
                    )

                if covars == "1":
                    covars = endogvars
                else:
                    covars = endogvars + "+" + covars
            else:
                fevars = fml_split[1]
                endogvars = None
                instruments = None
        elif len(fml_split) == 3:
            fevars = fml_split[1]
            endogvars, instruments = fml_split[2].split("~")

            # check if any of the instruments or endogeneous variables are also specified
            # as covariates
            if any(element in covars.split("+") for element in endogvars.split("+")):
                raise EndogVarsAsCovarsError(
                    "Endogeneous variables are specified as covariates in the first part of the three-part formula. This is not allowed."
                )

            if any(element in covars.split("+") for element in instruments.split("+")):
                raise InstrumentsAsCovarsError(
                    "Instruments are specified as covariates in the first part of the three-part formula. This is not allowed."
                )

            # add endogeneous variable to "covars" - yes, bad naming
            if covars == "1":
                covars = endogvars
            else:
                covars = endogvars + "+" + covars

        if endogvars is not None:
            if len(endogvars) > len(instruments):
                raise UnderDeterminedIVError(
                    "The IV system is underdetermined. Only fully determined systems are allowed. Please provide as many instruments as endogenous variables."
                )
            else:
                pass

        # Parse all individual formula components into lists
        self.depvars = depvars.split("+")
        self.covars = _unpack_fml(covars)
        self.fevars = _unpack_fml(fevars)
        # no fancy syntax for endogvars, instruments allowed
        self.endogvars = endogvars
        self.instruments = instruments

        # clean instruments
        if instruments is not None:
            self._is_iv = True
            # all rhs variables for the first stage (endog variable replaced with instrument)
            first_stage_covars_list = covars.split("+")
            first_stage_covars_list[
                first_stage_covars_list.index(endogvars)
            ] = instruments
            self.first_stage_covars_list = "+".join(first_stage_covars_list)
            self.covars_first_stage = _unpack_fml(self.first_stage_covars_list)
            self.depvars_first_stage = endogvars
        else:
            self._is_iv = False
            self.covars_first_stage = None
            self.depvars_first_stage = None

        # Pack the formula components back into strings
        self.covars_fml = _pack_to_fml(self.covars)
        self.fevars_fml = _pack_to_fml(self.fevars)
        if instruments is not None:
            self.covars_first_stage_fml = _pack_to_fml(self.covars_first_stage)
        else:
            self.covars_first_stage_fml = None

    def get_new_fml_dict(self, iv=False):
        """
        Get a nested dictionary of all formulas.

        Parameters:
            iv: bool (default: False)
                If True, the formulas for the first stage are returned. Otherwise, the formulas for the second stage are returned.
        Returns:
            fml_dict: dict
                A nested dictionary of all formulas. The dictionary has the following structure: first, a dictionary with the
                fixed effects combinations as keys. Then, for each fixed effect combination, a dictionary with the dependent variables
                as keys. Finally, for each dependent variable, a list of formulas as values.

                Here is an example:
                    fml = Y1 + Y2 ~ X1 + X2 | FE1 + FE2 is transformed into: {"FE1 + FE2": {"Y1": "Y2 ~X1+X2", "Y2":"X1+X2"}}
        """

        fml_dict = dict()

        for fevar in self.fevars_fml:
            res = dict()
            for depvar in self.depvars:
                res[depvar] = []
                if iv:
                    for covar in self.covars_first_stage_fml:
                        res[depvar].append(depvar + "~" + covar)
                else:
                    for covar in self.covars_fml:
                        res[depvar].append(depvar + "~" + covar)
            fml_dict[fevar] = res

        if iv:
            self._fml_dict_new_iv = fml_dict
        else:
            self._fml_dict_new = fml_dict


def _unpack_fml(x):
    """
    Given a formula string `x` - e.g. 'X1 + csw(X2, X3)' - , splits it into its constituent variables and their types (if any),
    and returns a dictionary containing the result. The dictionary has the following keys: 'constant', 'sw', 'sw0', 'csw'.
    The values are lists of variables of the respective type.

    Parameters:
    -----------
    x : str
        The formula string to unpack.

    Returns:
    --------
    res_s : dict
        A dictionary containing the unpacked formula. The dictionary has the following keys:
            - 'constant' : list of str
                The list of constant (i.e., non-switched) variables in the formula.
            - 'sw' : list of str
                The list of variables that have a regular switch (i.e., 'sw(var1, var2, ...)' notation) in the formula.
            - 'sw0' : list of str
                The list of variables that have a 'sw0(var1, var2, ..)' switch in the formula.
            - 'csw' : list of str or list of lists of str
                The list of variables that have a 'csw(var1, var2, ..)' switch in the formula.
                Each element in the list can be either a single variable string, or a list of variable strings
                if multiple variables are listed in the switch.
            - 'csw0' : list of str or list of lists of str
                The list of variables that have a 'csw0(var1,var2,...)' switch in the formula.
                Each element in the list can be either a single variable string, or a list of variable strings
                if multiple variables are listed in the switch.

    Raises:
    -------
    ValueError:
        If the switch type is not one of 'sw', 'sw0', 'csw', or 'csw0'.

    Example:
    --------
    >>> _unpack_fml('a+sw(b)+csw(x1,x2)+sw0(d)+csw0(y1,y2,y3)')
    {'constant': ['a'],
     'sw': ['b'],
     'csw': [['x1', 'x2']],
     'sw0': ['d'],
     'csw0': [['y1', 'y2', 'y3']]}
    """

    # Split the formula into its constituent variables
    var_split = x.split("+")

    res_s = dict()
    res_s["constant"] = []

    for var in var_split:
        # Check if this variable contains a switch
        varlist, sw_type = _find_sw(var)

        # If there's no switch, just add the variable to the list
        if sw_type is None:
            if _is_varying_slopes(var):
                varlist, sw_type = _transform_varying_slopes(var)
                for x in varlist.split("+"):
                    res_s["constant"].append(x)
            else:
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


def _pack_to_fml(unpacked):
    """
    Given a dictionary of "unpacked" formula variables, returns a string containing formulas. An "unpacked" formula is a
    deparsed formula that allows for multiple estimations.

    Parameters
    ----------
    unpacked : dict
        A dictionary of unpacked formula variables. The dictionary has the following keys:
            - 'constant' : list of str
                The list of constant (i.e., non-switched) variables in the formula.
            - 'sw' : list of str
                The list of variables that have a regular switch (i.e., 'sw(var1, var2, ...)' notation) in the formula.
            - 'sw0' : list of str
                The list of variables that have a 'sw0(var1, var2, ..)' switch in the formula.
            - 'csw' : list of str or list of lists of str
                The list of variables that have a 'csw(var1, var2, ..)' switch in the formula.
                Each element in the list can be either a single variable string, or a list of variable strings
                if multiple variables are listed in the switch.
            - 'csw0' : list of str or list of lists of str
    """

    res = dict()

    # add up all constant variables
    if "constant" in unpacked:
        res["constant"] = unpacked["constant"]
    else:
        res["constant"] = []

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

    if res["constant"]:
        const_fml = "+".join(res["constant"])
    else:
        const_fml = []

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
                const_fml + "+" + variable_fml[i]
                for i in range(len(variable_fml))
                if variable_fml[i] != "0"
            ]
            if variable_type in ["sw0", "csw0"]:
                fml_list = [const_fml] + fml_list
        else:
            fml_list = variable_fml
    else:
        if const_fml:
            fml_list = const_fml
        else:
            raise Exception("Not a valid formula provided.")

    if not isinstance(fml_list, list):
        fml_list = [fml_list]

    return fml_list


def _find_sw(x):
    """
    Search for matches in a string. Matches are either 'sw', 'sw0', 'csw', 'csw0'. If a match is found, returns a
    tuple containing a list of the elements found and the type of match. Otherwise, returns the original string and None.

    Args:
        x (str): The string to search for matches in.

    Returns:
        (list[str] or str, str or None): If any matches were found, returns a tuple containing
        a list of the elements found and the type of match (either 'sw', 'sw0', 'csw', or 'csw0').
        Otherwise, returns the original string and None.

    Example:
        _find_sw('sw(var1, var2)') -> (['var1', ' var2'], 'sw')
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


def _flatten_list(lst):
    """
    Flattens a list that may contain sublists.

    Args:
        lst (list): A list that may contain sublists.

    Returns:
        list: A flattened list with no sublists.

    Examples:
        >>> flatten_list([[1, 2, 3], 4, 5])
        [1, 2, 3, 4, 5]
        >>> flatten_list([1, 2, 3])
        [1, 2, 3]
    """

    flattened_list = []
    for i in lst:
        if isinstance(i, list):
            flattened_list.extend(_flatten_list(i))
        else:
            flattened_list.append(i)
    return flattened_list


def _check_duplicate_key(my_dict, key):
    """
    Checks if a key already exists in a dictionary. If it does, raises a DuplicateKeyError. Otherwise, does nothing.

    Args:
        my_dict (dict): The dictionary to check for duplicate keys.
        key (str): The key to check for in the dictionary.

    Returns:
        None
    """

    for key in ["sw", "csw", "sw0", "csw0"]:
        if key in my_dict:
            raise DuplicateKeyError(
                "Duplicate key found: "
                + key
                + ". Multiple estimation syntax can only be used once on the rhs of the two-sided formula."
            )
        else:
            None


def _is_varying_slopes(x):
    pattern = r"\[.*\]"
    match = re.search(pattern, x)
    if match:
        return True
    else:
        return False


def _transform_varying_slopes(x):
    parts = x.split("[")
    a = parts[0]
    b = parts[1].replace("]", "")
    transformed_string = f"{a}/{b}"
    return transformed_string, "varying_slopes"
