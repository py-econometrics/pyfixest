import pandas as pd
from formulaic.utils.stateful_transforms import stateful_transform


@stateful_transform
def fixed_effect_interactions(*args, _state=None, _metadata=None, _spec=None):
    data = pd.concat(args, axis=1)
    if "__id__" not in _state:
        print(_state)
        data["__id__"] = data.groupby(data.columns.tolist()).ngroup()
        _state["__id__"] = data.drop_duplicates()
        return data["__id__"]

    return data.merge(_state, on=data.columns, how="left")["__id__"]
