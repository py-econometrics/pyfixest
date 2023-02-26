def bunch_factory(attribute, columns):
    """
    Generates a special purpose Bunch class
    Parameters
    ----------
    attribute: str
        Attribute to access when splitting
    columns: List[str]
        List of names to use when splitting the columns of attribute
    Notes
    -----
    After the class is initialized as a Bunch, the columne of attribute
    are split so that Bunch has the keys in columns and
    bunch[column[i]] = bunch[attribute][:, i]
    """
    class FactoryBunch(Bunch):
        def __init__(self, *args, **kwargs):
            super(FactoryBunch, self).__init__(*args, **kwargs)
            if not hasattr(self, attribute):
                raise AttributeError('{0} is required and must be passed to '
                                     'the constructor'.format(attribute))
            for i, att in enumerate(columns):
                self[att] = getattr(self, attribute)[:, i]

    return FactoryBunch


ParamsTableTestBunch = bunch_factory('params_table', PARAM_LIST)