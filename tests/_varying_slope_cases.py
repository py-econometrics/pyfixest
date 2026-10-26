"""Varying-slope formulas shared by the parsing and R fixest suites.

`tests/test_formula_parse.py` checks that each formula parses to `effects`, and
`tests/test_varying_slopes_vs_fixest.py` checks that the parsed model matches
`fml_r` in R fixest.
"""

from __future__ import annotations

from typing import NamedTuple

# (levels, intercept, slopes) for each fixed effect, in model order.
Effects = tuple[tuple[str, bool, tuple[str, ...]], ...]


class SlopeCase(NamedTuple):
    fml: str
    fml_r: str
    effects: Effects
    # Cross with every vcov and weights option instead of only iid.
    full_matrix: bool = False


def _same(fml: str, effects: Effects, *, full_matrix: bool = False) -> SlopeCase:
    return SlopeCase(fml, fml, effects, full_matrix)


OLS_SLOPE_CASES = (
    _same("Y ~ X1 | f1[X2]", (("f1", True, ("X2",)),), full_matrix=True),
    _same("Y ~ X1 | f1[[X2]]", (("f1", False, ("X2",)),), full_matrix=True),
    _same(
        "Y ~ X1 | f1[[X2, Z1]] + f2",
        (("f1", False, ("X2", "Z1")), ("f2", True, ())),
        full_matrix=True,
    ),
    SlopeCase(
        "Y ~ X1 | f1:f2[X2]",
        "Y ~ X1 | f1^f2[X2]",
        (("f1:f2", True, ("X2",)),),
        full_matrix=True,
    ),
    _same("Y ~ 0 + X1 | f1[[X2]]", (("f1", False, ("X2",)),)),
    _same("Y ~ X1 | f1[X2, Z1] + f2", (("f1", True, ("X2", "Z1")), ("f2", True, ()))),
    _same(
        "Y ~ X1 | f1[[X2]] + f2[[X2]]",
        (("f1", False, ("X2",)), ("f2", False, ("X2",))),
    ),
    # Effects on the same levels merge into one effect.
    _same(
        "Y ~ X1 | f1 + f1[[X2]] + f2[[Z1]]",
        (("f1", True, ("X2",)), ("f2", False, ("Z1",))),
    ),
    _same("Y ~ X1 | f1[[X2]] + f1", (("f1", True, ("X2",)),)),
    _same("Y ~ X1 | f1[X2, X2]", (("f1", True, ("X2",)),)),
    _same("Y ~ X1 | f1[[X2, X2]]", (("f1", False, ("X2",)),)),
    _same("Y ~ X1 | f1[[X2]] + f1[[X2]]", (("f1", False, ("X2",)),)),
    _same("Y ~ X1 | f1[[X2]] + f1[Z1]", (("f1", True, ("X2", "Z1")),)),
    _same("Y ~ X1 | f1[X2] + f1[Z1]", (("f1", True, ("X2", "Z1")),)),
    _same(
        "Y ~ X1 | f1[[X2]] + f2 + f1[[Z1]]",
        (("f1", False, ("X2", "Z1")), ("f2", True, ())),
    ),
    # fixest keeps `f2^f1` separate from `f1^f2` and its solver does not
    # converge, so compare with fixest's spelling of the merged effect.
    SlopeCase(
        "Y ~ X1 | f1:f2[[X2]] + f2:f1",
        "Y ~ X1 | f1^f2[X2]",
        (("f1:f2", True, ("X2",)),),
    ),
    # Slopes are evaluated arithmetic, not expanded formula terms.
    SlopeCase(
        "Y ~ X1 | f1[X2**2]", "Y ~ X1 | f1[I(X2^2)]", (("f1", True, ("X2 ** 2",)),)
    ),
    SlopeCase(
        "Y ~ X1 | f1[np.exp(Z1)]",
        "Y ~ X1 | f1[exp(Z1)]",
        (("f1", True, ("np.exp(Z1)",)),),
    ),
    _same("Y ~ X1 | f1[X2 * Z1]", (("f1", True, ("X2 * Z1",)),)),
    _same("Y ~ X1 | f1[X2 + Z1]", (("f1", True, ("X2 + Z1",)),)),
    # A regressor equal to the slope covariate is collinear with the slopes.
    _same("Y ~ X2 + X1 | f1[X2]", (("f1", True, ("X2",)),)),
    _same("Y ~ X2 + X1 | f1[[X2]]", (("f1", False, ("X2",)),)),
)

IV_SLOPE_CASES = (
    SlopeCase(
        "Y ~ X2 + [X1 ~ Z1] | f1[Z2]",
        "Y ~ X2 | f1[Z2] | X1 ~ Z1",
        (("f1", True, ("Z2",)),),
        full_matrix=True,
    ),
    SlopeCase(
        "Y ~ X2 + [X1 ~ Z1] | f1[[Z2]] + f2",
        "Y ~ X2 | f1[[Z2]] + f2 | X1 ~ Z1",
        (("f1", False, ("Z2",)), ("f2", True, ())),
        full_matrix=True,
    ),
)

GLM_SLOPE_CASES = (
    OLS_SLOPE_CASES[0],
    OLS_SLOPE_CASES[1],
    OLS_SLOPE_CASES[2],
    OLS_SLOPE_CASES[3],
    _same("Y ~ X1 | f1[[X2]] + f1[Z1]", (("f1", True, ("X2", "Z1")),)),
)

# Every distinct formula, for checks that do not depend on the estimator.
ALL_SLOPE_CASES = tuple(
    {
        case.fml: case for case in OLS_SLOPE_CASES + IV_SLOPE_CASES + GLM_SLOPE_CASES
    }.values()
)
