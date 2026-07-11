"""Define and validate literal options shared by estimation APIs."""

from typing import Any, Literal, get_args

from pyfixest.typing import GlmFamily, WeightsType

PredictionType = Literal["response", "link"]
VcovTypeOptions = Literal["iid", "hetero", "HC1", "HC2", "HC3", "NW", "DK", "nid"]
WeightsTypeOptions = WeightsType
FixedRmOptions = Literal["singleton", "none"]
FamilyOptions = GlmFamily
SolverOptions = Literal[
    "np.linalg.lstsq",
    "np.linalg.solve",
    "scipy.linalg.solve",
    "scipy.sparse.linalg.lsqr",
]
PredictionErrorOptions = Literal["prediction"]
QuantregMethodOptions = Literal["fn", "pfn"]
QuantregMultiOptions = Literal["cfm1", "cfm2"]


def _validate_literal_argument(arg: Any, literal: Any) -> None:
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
    TypeError
        If `literal` does not have valid types.
    ValueError
        If `arg` is not one of the valid types defined by `literal`.
    """
    valid_types = get_args(literal)

    if len(valid_types) < 1:
        raise TypeError(
            f"{literal} must be a Literal[...] type argument with least one type"
        )

    if arg not in valid_types:
        raise ValueError(f"Invalid argument. Expecting one of {valid_types}. Got {arg}")
