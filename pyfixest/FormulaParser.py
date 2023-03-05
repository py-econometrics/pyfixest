import re

class FixestFormulaParser:


    def __init__(self, fml):

        '''
        Split Formula into individual lists - depvars, covars, fevars.
        For covars and fevars, where multiple estimation is supported,
        move the sublist to the beginning of the respective list
        '''

        fml = "".join(fml.split())
        fml_split = fml.split('|')
        depvars, covars = fml_split[0].split("~")
        fevars = fml_split[1]

        self.depvars = depvars.split("+")
        self.covars = _unpack_fml(covars)
        self.fevars = _unpack_fml(fevars)

        self.covars_fml = _pack_to_fml(self.covars)
        self.fevars_fml = _pack_to_fml(self.fevars)

    def get_fml_dict(self):

        '''
        a dictionary of all fevars & formula
        without fevars

        {"fe1+fe2": ['Y1 ~ X', 'Y2~X'], "fe1+fe3": ['Y1 ~ X', 'Y2~X']}

        '''

        self.fml_dict = dict()
        for fevar in self.fevars_fml:
            res = []
            for depvar in self.depvars:
               for covar in self.covars_fml:
                   res.append(depvar + '~' + covar)
            self.fml_dict[fevar] = res


    def get_var_dict(self):

        '''
        a dictionary of all fevars and list of covars and depvars used in regression
        with those fevars

        e.g. returns

        {"fe1+fe2": ['Y1', 'X1', 'X2'], "fe1+fe3": ['Y1', 'X1', 'X2']}

        '''

        self.var_dict = dict()
        for fevar in self.fevars_fml:
              self.var_dict[fevar] = _flatten_list(self.depvars) + _flatten_list(self.covars)






def _unpack_fml(x):

    '''
    Find all variables in a formula and unpack.

    Examples:
        var: "a + sw(b, c)" -> ['a', ['b', 'c']]
        var = "a + csw(b, c)" -> ['a', ['b', 'b + c']]
        var = "a + csw0(b,c) + d" -> ['a', ['b', 'b + c'], 'd']

    '''

    res_s = []
    var_split = x.split("+")

    for x in var_split:

        #if isinstance(x, list) & len(x) == 1:
        #    x = x[0]

        varlist, sw_type = _find_sw(x)
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

    res_s.sort(key=lambda x: 0 if isinstance(x, list) else 1)
      
    return res_s


def _pack_to_fml(unpacked):

    '''
    Args:
        x (list): contains str or list of str
    Returns:
        A list.

    based on output from 'unpack_fml' - which might e.g. look like
    this [['0', 'x4', ' x5', ' x6'], 'x1 ', ' x2 ', ' x3'] -
    get a lit of formulas back:
    ['x1 + x2 + x3+0', 'x1 + x2 + x3+x4', 'x1 + x2 + x3+ x5', 'x1 + x2 + x3+ x6']
    '''

    # get the fixed variables
    constant_list = [x for x in unpacked if not isinstance(x, list)]
    sw_list = [x for x in unpacked if isinstance(x, list)]

    #if len(sw_list) == 1 and isinstance(sw_list[0], list):
    #    sw_list = sw_list[0]

    res = []
    if sw_list != [] and constant_list != []:
        constant_list =  "+".join(constant_list)
        for x in sw_list:
            res.append(constant_list + "+" + x)
    elif sw_list != []:
        res = sw_list[0]
    else:
        res = "+".join(constant_list)

    if isinstance(res, str):
        res = [res]


    return res



def _find_sw(x):

    '''
    for a given string x, find all elements within 'type'
    enbracketed in supported formula syntax sugar,
    e.g. 'var1, var2' in 'sw(var1, var2)'
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


def _flatten_list(lst):

    '''
    flatten a list with sublist

    Example:
        _flatten_list([[1, 2, 3], 4, 5]) -> [1, 2, 3, 4, 5]
    '''

    flattened_list = []
    for i in lst:
        if isinstance(i, list):
            flattened_list.extend(_flatten_list(i))
        else:
            flattened_list.append(i)
    return flattened_list



