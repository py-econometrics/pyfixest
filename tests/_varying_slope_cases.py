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
    full_inference_grid: bool = False


def _same_formula(
    fml: str, effects: Effects, *, full_inference_grid: bool = False
) -> SlopeCase:
    return SlopeCase(fml, fml, effects, full_inference_grid)


OLS_SLOPE_CASES = (
    _same_formula(
        "Y ~ X1 | f1[X2]", (("f1", True, ("X2",)),), full_inference_grid=True
    ),
    _same_formula(
        "Y ~ X1 | f1[[X2]]", (("f1", False, ("X2",)),), full_inference_grid=True
    ),
    _same_formula(
        "Y ~ X1 | f1[[X2, Z1]] + f2",
        (("f1", False, ("X2", "Z1")), ("f2", True, ())),
        full_inference_grid=True,
    ),
    SlopeCase(
        "Y ~ X1 | f1:f2[X2]",
        "Y ~ X1 | f1^f2[X2]",
        (("f1:f2", True, ("X2",)),),
        full_inference_grid=True,
    ),
    _same_formula("Y ~ 0 + X1 | f1[[X2]]", (("f1", False, ("X2",)),)),
    _same_formula("Y ~ 1 | f1[X2]", (("f1", True, ("X2",)),)),
    _same_formula("Y ~ X1 | f1[[X2, Z1]]", (("f1", False, ("X2", "Z1")),)),
    _same_formula(
        "Y ~ X1 | f1[X2, Z1] + f2", (("f1", True, ("X2", "Z1")), ("f2", True, ()))
    ),
    _same_formula(
        "Y ~ X1 | f1[[X2]] + f2[[X2]]",
        (("f1", False, ("X2",)), ("f2", False, ("X2",))),
    ),
    SlopeCase(
        "Y ~ X1 | f1:f2[[X2]]",
        "Y ~ X1 | f1^f2[[X2]]",
        (("f1:f2", False, ("X2",)),),
    ),
    # Effects on the same levels merge into one effect.
    _same_formula(
        "Y ~ X1 | f1 + f1[[X2]] + f2[[Z1]]",
        (("f1", True, ("X2",)), ("f2", False, ("Z1",))),
    ),
    _same_formula("Y ~ X1 | f1[[X2]] + f1", (("f1", True, ("X2",)),)),
    _same_formula("Y ~ X1 | f1[X2, X2]", (("f1", True, ("X2",)),)),
    _same_formula("Y ~ X1 | f1[[X2, X2]]", (("f1", False, ("X2",)),)),
    # A slope repeated across terms is kept once.
    SlopeCase(
        "Y ~ X1 | f1[[Z1]] + f1[X2, Z1]",
        "Y ~ X1 | f1[Z1, X2]",
        (("f1", True, ("Z1", "X2")),),
    ),
    _same_formula("Y ~ X1 | f1[[X2]] + f1[Z1]", (("f1", True, ("X2", "Z1")),)),
    _same_formula("Y ~ X1 | f1[X2] + f1[Z1]", (("f1", True, ("X2", "Z1")),)),
    _same_formula(
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
    _same_formula("Y ~ X1 | f1[X2 * Z1]", (("f1", True, ("X2 * Z1",)),)),
    _same_formula("Y ~ X1 | f1[X2 + Z1]", (("f1", True, ("X2 + Z1",)),)),
    # Without an intercept, shifting the slope covariate changes the model.
    _same_formula("Y ~ X1 | f1[[X2 + 1]]", (("f1", False, ("X2 + 1",)),)),
    # One column enters slopes of two effects, once transformed.
    SlopeCase(
        "Y ~ X1 | f1[Z1] + f2[[X2, Z1**2]]",
        "Y ~ X1 | f1[Z1] + f2[[X2, I(Z1^2)]]",
        (("f1", True, ("Z1",)), ("f2", False, ("X2", "Z1 ** 2"))),
    ),
    # A regressor equal to the slope covariate is collinear with the slopes.
    _same_formula("Y ~ X2 + X1 | f1[X2]", (("f1", True, ("X2",)),)),
    _same_formula("Y ~ X2 + X1 | f1[[X2]]", (("f1", False, ("X2",)),)),
)

# fixest drops the intercept of slope-only IV models from its parameter count
# and residuals; see the compatibility ledger for the explicit 2SLS comparison.
IV_SLOPE_CASES = (
    SlopeCase(
        "Y ~ X2 + [X1 ~ Z1] | f1[Z2]",
        "Y ~ X2 | f1[Z2] | X1 ~ Z1",
        (("f1", True, ("Z2",)),),
        full_inference_grid=True,
    ),
    SlopeCase(
        "Y ~ X2 + [X1 ~ Z1] | f1[[Z2]] + f2",
        "Y ~ X2 | f1[[Z2]] + f2 | X1 ~ Z1",
        (("f1", False, ("Z2",)), ("f2", True, ())),
        full_inference_grid=True,
    ),
    SlopeCase(
        "Y ~ 0 + X2 + [X1 ~ Z1] | f1[[Z2]]",
        "Y ~ 0 + X2 | f1[[Z2]] | X1 ~ Z1",
        (("f1", False, ("Z2",)),),
    ),
)

GLM_SLOPE_CASES = (
    OLS_SLOPE_CASES[0],
    OLS_SLOPE_CASES[1],
    OLS_SLOPE_CASES[2],
    OLS_SLOPE_CASES[3],
    _same_formula(
        "Y ~ X1 | f1[[X2]] + f1[Z1]",
        (("f1", True, ("X2", "Z1")),),
        full_inference_grid=True,
    ),
    _same_formula(
        "Y ~ 0 + X1 | f1[[X2]]", (("f1", False, ("X2",)),), full_inference_grid=True
    ),
    _same_formula(
        "Y ~ X1 | f1[X2] + f2[Z1]", (("f1", True, ("X2",)), ("f2", True, ("Z1",)))
    ),
    SlopeCase(
        "Y ~ X1 | f1:f2[[X2]]",
        "Y ~ X1 | f1^f2[[X2]]",
        (("f1:f2", False, ("X2",)),),
    ),
    SlopeCase(
        "Y ~ X1 | f1:f2[[X2, Z1]]",
        "Y ~ X1 | f1^f2[[X2, Z1]]",
        (("f1:f2", False, ("X2", "Z1")),),
    ),
)

# Every distinct formula, for checks that do not depend on the estimator.
ALL_SLOPE_CASES = tuple(
    {
        case.fml: case for case in OLS_SLOPE_CASES + IV_SLOPE_CASES + GLM_SLOPE_CASES
    }.values()
)
