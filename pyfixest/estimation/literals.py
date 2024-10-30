from typing import Literal, _LiteralGenericAlias, get_args, Any

PredictionType = Literal["response", "link"]
VcovTypeOptions = Literal["iid", "hetero", "HC1", "HC2", "HC3"]
WeightsTypeOptions = Literal["aweights", "fweights"]
FixedRmOptions = Literal["singleton", "none"]
SolverOptions = Literal["np.linalg.solve", "np.linalg.lstsq"]


def _validate_literal_argument(arg: Any, literal: _LiteralGenericAlias) -> None:
    """
    Validate if the given argument matches one of the allowed literal types.

    This function checks whether the provided `arg` is among the valid types
    returned by `get_args(literal)`. If not, it raises a ValueError with an
    appropriate error message.

    Parameters
    ----------
    arg : Any
        The argument to validate.
    literal : _LiteralGenericAlias
        A Literal type that defines the allowed values for `arg`.

    Raises
    ------
    ValueError
        If `arg` is not one of the valid types defined by `literal`.
    """
    if type(literal) is _LiteralGenericAlias:
        valid_types = get_args(literal)
    else:
        raise TypeError(f"{literal} must be of Literal[...] type")

    if arg not in valid_types:
        raise ValueError(f"Invalid argument. Expecting one of {valid_types}. Got {arg}")