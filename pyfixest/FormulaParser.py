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

        #fml =' Y + Y2 ~  i(X1, X2) |csw0(X3, X4)'

        # Clean up the formula string
        fml = "".join(fml.split())

        # Split the formula string into its components
        fml_split = fml.split('|')
        depvars, covars = fml_split[0].split("~")

        if len(fml_split) > 1:
            fevars = fml_split[1]
        else:
            fevars = "0"

        # Parse all individual formula components into lists
        self.depvars = depvars.split("+")
        self.covars = _unpack_fml(covars)
        self.fevars = _unpack_fml(fevars)

        if self.covars.get("^") is not None:
            raise ValueError("Please use 'i()' or ':' syntax to interact covariates.")

        if self.fevars.get("i") is not None:
            raise ValueError("Please use '^' to interact fixed effects.")

        if self.covars.get("i") is not None:
            self.ivars = dict()
            i_split = self.covars.get("i")[-1].split("=")
            if len(i_split) > 1: 
                ref = self.covars.get("i")[-1].split("=")[1]
                ivar_list = self.covars.get("i")[:-1]
                self.covars["i"] = self.covars.get("i")[:-1]
            else: 
                ref = None
                ivar_list = self.covars.get("i")

            self.ivars[ref] = ivar_list
        
        else:
            self.ivars = None


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
              self.var_dict[fevar] = _flatten_list(self.depvars) + _flatten_list(list(self.covars.values()))



def _unpack_fml(x):

    '''
    Given a formula string `x` - e.g. 'X1 + csw(X2, X3)' - , splits it into its constituent variables and their types (if any),
    and returns a dictionary containing the result.

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
    '''


    # Split the formula into its constituent variables
    var_split = x.split("+")

    res_s = dict()
    res_s['constant'] = []

    for var in var_split:

        # Check if this variable contains a switch
        varlist, sw_type = _find_sw(var)

        # If there's no switch, just add the variable to the list
        if sw_type is None:
            res_s['constant'].append(var)

        # If there's a switch, unpack it and add it to the list
        else:
            if sw_type in ['sw', 'sw0', 'csw', 'csw0', 'i']:
                res_s[sw_type] = varlist
            else:
                raise ValueError("Unsupported switch type")

    # Sort the list by type (strings first, then lists)
    #res_s.sort(key=lambda x: 0 if isinstance(x, str) else 1)

    return res_s




def _pack_to_fml(unpacked):
    """
    """

    res = dict()

    # add up all constant variables
    if 'constant' in unpacked:
         res['constant'] = unpacked['constant']
    else:
        res['constant'] = []

    if 'i' in unpacked:
       if res['constant']:
           res['constant'] =  res['constant'] + [":".join(unpacked['i'])]
       else:
           res['constant'] = [":".join(unpacked['i'])]

    # add up all variable constants (only required for csw)
    if "csw" in unpacked:
        res['variable'] = unpacked['csw']
        variable_type = "csw"
    elif "csw0" in unpacked:
        res['variable'] = unpacked['csw0']
        variable_type = "csw0"
    elif "sw" in unpacked:
        res['variable'] = unpacked['sw']
        variable_type = "sw"
    elif "sw0" in unpacked:
        res['variable'] = unpacked['sw0']
        variable_type = "sw0"
    else:
        res['variable'] = []
        variable_type = None

    if res['constant']:
        const_fml = "+".join(res['constant'])
    else:
        const_fml = []

    variable_fml = []
    if res['variable']:
        if variable_type in ['csw', 'csw0']:
            variable_fml = [ "+".join(res['variable'][:i+1]) for i in range(len(res['variable']))]
        else:
            variable_fml = [res['variable'][i] for i in range(len(res['variable']))]
        if variable_type in ['sw0', 'csw0']:
            variable_fml = ['0'] + variable_fml


    fml_list = []
    if variable_fml:
        if const_fml:
            fml_list = [ const_fml + "+" + variable_fml[i] for i in range(len(variable_fml)) if variable_fml[i] != "0"]
            if variable_type in ['sw0', 'csw0']:
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
    i_match = re.findall(r"i\((.*?)\)", x)


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

    elif i_match:
        return i_match[0].split(","), "i"

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



def _check_unique_key(dict, x):

    '''
    check that sw, csw, sw0, csw0 are only used once in dict
    '''

    unique = True
    for key in dict:
        if key == x:
            continue
        if key == x:
            unique = False
            break

    if not unique:
        raise Exception(f"The key '{key_to_check}' is unique")
