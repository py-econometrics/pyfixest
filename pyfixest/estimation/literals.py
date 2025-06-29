from typing import Any, Literal, get_args

PredictionType = Literal["response", "link"]
VcovTypeOptions = Literal["iid", "hetero", "HC1", "HC2", "HC3", "nid", "iid"]
WeightsTypeOptions = Literal["aweights", "fweights"]
FixedRmOptions = Literal["singleton", "none"]
SolverOptions = Literal[
    "np.linalg.lstsq",
    "np.linalg.solve",
    "scipy.linalg.solve",
    "scipy.sparse.linalg.lsqr",
    "jax",
]
DemeanerBackendOptions = Literal["numba", "jax", "rust"]
PredictionErrorOptions = Literal["prediction"]
QuantregMethodOptions = Literal["fn", "pfn", "pfn_process"]
QuantregMultiOptions = Literal["cfm1", "cfm2", "none"]


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
