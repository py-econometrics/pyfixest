from typing import Literal, Any, get_args

prediction_type = Literal["response", "link"]
vcov_type_options = Literal["iid", "hetero", "HC1", "HC2", "HC3"]
weights_type_options = Literal["aweights", "fweights"]
fixef_rm_options = Literal["singleton", "none"]
solver_options = Literal["np.linalg.solve", "np.linalg.lstsq"]


def validate_literal_argument(arg: Any, literal: Literal) -> None:
    valid_types = get_args(literal)

    if arg not in valid_types:
        raise ValueError(
            f"Invalid prediction type. Expecting one of {valid_types}.\
                Got {arg}"
        )
