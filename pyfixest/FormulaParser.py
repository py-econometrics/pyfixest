import re

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
        """

        # Clean up the formula string
        fml = "".join(fml.split())

        # Split the formula string into its components
        fml_split = fml.split('|')
        depvars, covars = fml_split[0].split("~")

        if len(fml_split) > 1:
            fevars = fml_split[1]
        else:
            fevars = "0"

        # Parse the formula components into lists
        self.depvars = depvars.split("+")
        self.covars = _unpack_fml(covars)
        self.fevars = _unpack_fml(fevars)

        # Pack the formula components back into strings
        self.covars_fml = _pack_to_fml(self.covars)
        self.fevars_fml = _pack_to_fml(self.fevars)

    def get_fml_dict(self):

        """
        Returns a dictionary of all fevars & formula without fevars.

        Returns:
            dict: A dictionary of the form {"fe1+fe2": ['Y1 ~ X', 'Y2~X'], "fe1+fe3": ['Y1 ~ X', 'Y2~X']} where
            the keys are the fixed effect variable combinations and the values are lists of formula strings
            that do not include the fixed effect variables.
        """

        self.fml_dict = dict()
        for fevar in self.fevars_fml:
            res = []
            for depvar in self.depvars:
               for covar in self.covars_fml:
                   res.append(depvar + '~' + covar)
            self.fml_dict[fevar] = res


    def get_var_dict(self):

        """
        Create a dictionary of all fevars and list of covars and depvars used in regression with those fevars.

        Returns:
            dict: A dictionary of the form {"fe1+fe2": ['Y1', 'X1', 'X2'], "fe1+fe3": ['Y1', 'X1', 'X2']} where
            the keys are the fixed effect variable combinations and the values are lists of variables
            (dependent variables and covariates) used in the regression with those fixed effect variables.

        """
        self.var_dict = dict()
        for fevar in self.fevars_fml:
              self.var_dict[fevar] = _flatten_list(self.depvars) + _flatten_list(self.covars)



def _unpack_fml(x):
    '''
    Find all variables in a formula and unpack.

    Args:
        x (str): A formula string.

    Returns:
        list: A list of strings and/or lists of strings representing the unpacked formula.

    Examples:
        var: "a + sw(b, c)" -> ['a', ['b', 'c']]
        var = "a + csw(b, c)" -> ['a', ['b', 'b + c']]
        var = "a + csw0(b,c) + d" -> ['a', ['b', 'b + c'], 'd']
    '''

    # Split the formula into its constituent variables
    var_split = x.split("+")

    res_s = []
    for var in var_split:

        # Check if this variable contains a switch
        varlist, sw_type = _find_sw(var)

        # If there's no switch, just add the variable to the list
        if sw_type is None:
            res_s.append(var)

        # If there's a switch, unpack it and add it to the list
        else:
            if sw_type == "sw":
                res_s.append(varlist)
            elif sw_type == "sw0":
                res_s.append(['0'] + varlist)
            elif sw_type in ["csw", "csw0"]:
                varlist = ["+".join(varlist[:i+1]) for i in range(len(varlist))]
                if sw_type == 'csw0':
                    res_s.append(['0'] + varlist)
                else:
                    res_s.append(varlist)
            else:
                raise ValueError("Unsupported switch type")

    # Sort the list by type (strings first, then lists)
    res_s.sort(key=lambda x: 0 if isinstance(x, str) else 1)

    return res_s




def _pack_to_fml(unpacked):
    """
    Given a list of strings and/or lists of strings, generates a list of formulas by concatenating
    each element of the list with the elements of the other lists.

    Args:
        unpacked (list): contains strings or lists of strings

    Returns:
        A list of formulas.

    Example:
        _pack_to_fml([['0', 'x4', ' x5', ' x6'], 'x1 ', ' x2 ', ' x3']) ->
        ['x1 + x2 + x3+0', 'x1 + x2 + x3+x4', 'x1 + x2 + x3+ x5', 'x1 + x2 + x3+ x6']
    """

    # Separate the fixed variables from the switch variables
    constant_list = [x for x in unpacked if not isinstance(x, list)]
    sw_list = [x for x in unpacked if isinstance(x, list)]

    # Generate the formulas by concatenating the constant list with each element of the switch list
    res = []
    if sw_list and constant_list:
        constant_list =  "+".join(constant_list)
        for x in sw_list:
            res.append(constant_list + "+" + x)
    elif sw_list:
        res = sw_list[0]
    else:
        res = "+".join(constant_list)

    # Return the result as a list, even if it's a single formula
    if isinstance(res, str):
        res = [res]

    return res


def _find_sw(x):
    """
    Search for matches in a string.

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



