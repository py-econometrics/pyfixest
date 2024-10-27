from typing import Any, Literal, get_args

prediction_type = Literal["response", "link"]
vcov_type_options = Literal["iid", "hetero", "HC1", "HC2", "HC3"]
weights_type_options = Literal["aweights", "fweights"]
fixef_rm_options = Literal["singleton", "none"]
solver_options = Literal["np.linalg.solve", "np.linalg.lstsq"]


def validate_literal_argument(arg: Any, literal: Any) -> None:
    """
    Validate if the given argument matches one of the allowed literal types.

    This function checks whether the provided `arg` is among the valid types
    returned by `get_args(literal)`. If not, it raises a ValueError with an
    appropriate error message.

    Parameters
    ----------
    arg : Any
        The argument to validate.
    literal : Any
        A Literal type that defines the allowed values for `arg`.

    Raises
    ------
    ValueError
        If `arg` is not one of the valid types defined by `literal`.
    """
    valid_types = get_args(literal)

    if arg not in valid_types:
        raise ValueError(
            f"Invalid prediction type. Expecting one of {valid_types}.\
                Got {arg}"
        )
