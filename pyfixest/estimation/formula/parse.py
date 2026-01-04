import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from formulaic.parser import DefaultFormulaParser

from pyfixest.errors import (
    DuplicateKeyError,
    EndogVarsAsCovarsError,
    FormulaSyntaxError,
    InstrumentsAsCovarsError,
    UnderDeterminedIVError,
)
from pyfixest.estimation.formula import FORMULAIC_FEATURE_FLAG


class _MultipleEstimationType(StrEnum):
    # See https://lrberge.github.io/fixest/reference/stepwise.html
    sw = "sequential stepwise"
    csw = "cumulative stepwise"
    sw0 = "sequential stepwise with zero step"
    csw0 = "cumulative stepwise with zero step"


@dataclass(kw_only=True, frozen=True)
class _MultipleEstimation:
    constant: list[str]
    variable: list[str]
    kind: _MultipleEstimationType | None = None

    @property
    def is_multiple(self) -> bool:
        return self.kind is not None

    @property
    def steps(self) -> list[str]:
        if self.kind is None or self.kind.name.endswith("0"):
            # Add zero step
            estimation_steps = ["+".join(self.constant) if self.constant else "0"]
        else:
            estimation_steps = []
        if self.kind is not None and self.kind.name.startswith("sw"):
            # Sequential stepwise estimation
            estimation_steps.extend(
                ["+".join([*self.constant, v]) for v in self.variable]
            )
        elif self.kind is not None and self.kind.name.startswith("csw"):
            # Cumulative stepwise estimation
            cumulative_slice: list[list[str]] = [
                self.variable[: i + 1] for i, _ in enumerate(self.variable)
            ]
            estimation_steps.extend(
                ["+".join(self.constant + v) for v in cumulative_slice]
            )
        return estimation_steps


@dataclass(kw_only=False, frozen=True)
class Formula:
    """
    A class representing a fixest model formula.

    Attributes
    ----------
    dependent : str
        The dependent variable.
    independent : str
        The independent variables for the second stage, separated by '+'.
        For IV regressions, this includes both exogenous covariates and the
        endogenous variable.
    fixed_effects : str | None
        Fixed effect variables, separated by '+'. None if no fixed effects.
    endogenous : str | None
        The endogenous variable in IV regression. None for OLS.
    instruments : str | None
        Instrumental variables for the endogenous variable, separated by '+'.
        None for OLS.
    intercept : bool
        Whether to include an intercept in the model.
    """

    dependent: str
    independent: str
    fixed_effects: str | None = None
    endogenous: str | None = None
    instruments: str | None = None
    intercept: bool = True

    @property
    def fml(self) -> str:
        """
        Reconstruct the full formula string from its components.

        Returns
        -------
        str
            The complete formula string in fixest format.
        """
        independent = self.independent
        if not self.intercept:
            independent = f"{independent}-1"
        formula = f"{self.dependent}~{independent}"
        if self.fixed_effects is not None:
            formula = f"{formula}|{self.fixed_effects}"
        if self.endogenous is not None and self.instruments is not None:
            formula = f"{formula}|{self.endogenous}~{self.instruments}"
        return formula

    @property
    def first_stage(self) -> str | None:
        """
        Return the first stage formula for IV regression.

        Note: Fixed effects are NOT included in this formula. This is intentional
        because this property is used by `model_matrix.py` to build model matrices
        via formulaic, where fixed effects are handled separately (encoded as
        integers and passed via a separate 'fe' key). The pyfixest `|` syntax for
        fixed effects is not compatible with formulaic's formula parsing.

        For contexts requiring the full formula with fixed effects (e.g., when
        passing to `feols()`), fixed effects must be appended manually.

        Returns
        -------
        str | None
            The first stage formula, or None if not an IV regression.
        """
        if self.endogenous is None or self.instruments is None:
            return None
        independent = f"{self.instruments}+{self.independent}-{self.endogenous}"
        if not self.intercept:
            independent = f"{independent}-1"
        return f"{self.endogenous}~{independent}"

    @property
    def second_stage(self) -> str:
        """
        Return the second stage formula for model matrix creation.

        Note: Fixed effects are NOT included in this formula. This is intentional
        because this property is used by `model_matrix.py` to build model matrices
        via formulaic, where fixed effects are handled separately (encoded as
        integers and passed via a separate 'fe' key, then absorbed via demeaning).
        The pyfixest `|` syntax for fixed effects is not compatible with formulaic's
        formula parsing.

        Returns
        -------
        str
            The second stage formula.
        """
        independent = f"{self.independent}"
        if not self.intercept:
            independent = f"{independent}-1"
        if (
            FORMULAIC_FEATURE_FLAG is DefaultFormulaParser.FeatureFlags.ALL
            and self.endogenous is not None
            and self.instruments is not None
        ):
            independent = f"{independent}+[{self.endogenous}~{self.instruments}]"
        return f"{self.dependent}~{independent}"


@dataclass(kw_only=True, frozen=True)
class ParsedFormula:
    """
    A class representing a parsed formula string.

    Attributes
    ----------
    formula: str
        The raw formula string.
    dependent: list[str]
        The dependent variables.
    independent: _MultipleEstimation
        The independent variables.
    fixed_effects: Optional[_MultipleEstimation]
        The fixed effect variables included.
    endogenous: Optional[list[str]]
        The endogenous variables.
    instruments: Optional[list[str]]
        The instrumental variables for the endogenous variables.
    """

    formula: str
    dependent: list[str]
    independent: _MultipleEstimation
    fixed_effects: _MultipleEstimation | None = None
    endogenous: list[str] | None = None
    instruments: list[str] | None = None
    intercept: bool = True

    def __post_init__(self):
        if self.is_multiple and self.is_iv:
            raise NotImplementedError(
                "Multiple Estimations is currently not supported with IV. "
                "This is mostly due to insufficient testing and will be possible with a future release of PyFixest."
            )

    @property
    def is_multiple(self) -> bool:
        """
        Check if the formula specifies multiple estimations.

        Returns
        -------
        bool
            True if the formula includes multiple dependent variables, stepwise
            specifications, or multiple fixed effects specifications.
        """
        return (
            (len(self.dependent) > 1)
            or self.independent.is_multiple
            or (self.fixed_effects is not None and self.fixed_effects.is_multiple)
        )

    @property
    def is_fixed_effects(self) -> bool:
        """
        Check if the formula includes fixed effects.

        Returns
        -------
        bool
            True if fixed effects are specified in the formula.
        """
        return self.fixed_effects is not None

    @property
    def is_iv(self) -> bool:
        """
        Check if the formula specifies an instrumental variables regression.

        Returns
        -------
        bool
            True if endogenous variables and instruments are specified.
        """
        return self.endogenous is not None

    def _collect_formula_kwargs(self) -> dict[str, list[str]]:
        kwargs: dict[str, list[str]] = {
            "dependent": self.dependent,
            "independent": self.independent.steps,
        }
        if self.fixed_effects is not None:
            kwargs.update({"fixed_effects": self.fixed_effects.steps})
        if self.endogenous is not None:
            kwargs.update({"endogenous": self.endogenous})
        if self.instruments is not None:
            kwargs.update({"instruments": self.instruments})
        return kwargs

    @property
    def specifications(self) -> dict[str | None, list[Formula]]:
        """
        Generate all formula specifications from stepwise syntax.

        For multiple estimation formulas (using sw, csw, sw0, csw0), this expands
        the specification into individual Formula objects. Results are grouped by
        their fixed effects specification.

        Returns
        -------
        dict[str | None, list[Formula]]
            Dictionary mapping fixed effects specifications to lists of Formula objects.
            The key is the fixed effects string, or None if no fixed effects.
        """
        # Get formulas by group of fixed effects
        estimations: defaultdict[str | None, list[Formula]] = defaultdict(list[Formula])
        dict_of_lists = self._collect_formula_kwargs()
        list_of_kwargs = [
            dict(zip(dict_of_lists.keys(), values))
            for values in itertools.product(*dict_of_lists.values())
        ]
        for kwargs in list_of_kwargs:
            if kwargs.get("fixed_effects") == "0":
                # Encode no fixed effects by `None`
                kwargs.pop("fixed_effects")
            formula = Formula(intercept=self.intercept, **kwargs)
            estimations[formula.fixed_effects].append(formula)
        return estimations


@dataclass(frozen=True)
class _Pattern:
    parts: re.Pattern = re.compile(r"\s*\|\s*")
    dependence: re.Pattern = re.compile(r"\s*~\s*")
    variables: re.Pattern = re.compile(r"\s*\+\s*")
    args: re.Pattern = re.compile(r"\s*,\s*")
    multiple_estimation: re.Pattern = re.compile(
        rf"(?P<key>{'|'.join(e.name for e in _MultipleEstimationType)})\((?P<variables>.*?)\)"
    )


def _parse_parts(formula: str) -> tuple[str, list[str]]:
    """
    Parse parts of a one- to three-sided formula string of the form "`dependent ~ independent | fixed effects | endogenous ~ instruments`".

    Parameters
    ----------
    formula: str
        The three sided formula string.

    Returns
    -------
        main_part: str
        other_parts: list[str]
    """
    max_parts: Final[int] = 3
    min_tildes: Final[int] = 1
    max_tildes: Final[int] = 2

    parts = re.split(_Pattern.parts, formula.strip())
    if len(parts) > max_parts:
        raise FormulaSyntaxError(
            f"Formula can have at most {max_parts} parts `dependent ~ independent | fixed effects | endogenous ~ instruments`, "
            f"received {len(parts)}: {formula}"
        )
    n_tildes_per_part: list[int] = [part.count("~") for part in parts]
    if max(n_tildes_per_part) > 1:
        raise FormulaSyntaxError(
            f"A formula part can contain at most 1 `~`: {[part for part, n_tildes in zip(parts, n_tildes_per_part) if n_tildes > 1]}"
        )
    elif sum(n_tildes_per_part) < min_tildes:
        raise FormulaSyntaxError(f"Formula string must have at least {min_tildes} `~`.")
    elif sum(n_tildes_per_part) > max_tildes:
        raise FormulaSyntaxError(
            f"Formula string can have at most {max_tildes} `~`: in the main part and optionally in an instrumental variable part."
        )
    main_part = parts.pop(0)
    return main_part, parts


def _parse_dependent_independent(part: str) -> tuple[list[str], list[str]]:
    if "~" not in part:
        raise FormulaSyntaxError(
            f"Expect formula of form `dependent ~ independent`, received {part}"
        )
    dependent, independent = (
        re.split(_Pattern.variables, variables)
        for variables in re.split(_Pattern.dependence, string=part)
    )
    return dependent, independent


def _parse_fixed_effects(parts: list[str]) -> list[str] | None:
    part_fe: str | None = next((part for part in parts if "~" not in part), None)
    if part_fe is None:
        return None
    else:
        return re.split(_Pattern.variables, part_fe)


def _parse_instrumental_variable(
    parts: list[str],
    independent: list[str],
) -> tuple[list[str], list[str]] | tuple[None, None]:
    """
    Parse non-main parts of formula for presence of instrumental variable (IV) regressions.
    IV regressions are identified as the non-main formula part containing a `~`.

    Parameters
    ----------
    parts: list[str]
        Non-main parts of formula string.
    independent: list[str]
        Independent variables of main part of formula string.

    Returns
    -------
    endogenous, instruments: tuple[list[str], list[str]] | None

    """
    part_iv: str | None = next((part for part in parts if "~" in part), None)
    if part_iv is None:
        return None, None
    else:
        endogenous, instruments = _parse_dependent_independent(part_iv)
        endogenous_are_covariates = [
            variable for variable in endogenous if variable in independent
        ]
        if endogenous_are_covariates:
            raise EndogVarsAsCovarsError(
                f"Endogeneous variables specified as covariates: {endogenous_are_covariates}"
            )
        instruments_are_covariates = [
            variable for variable in instruments if variable in independent
        ]
        if instruments_are_covariates:
            raise InstrumentsAsCovarsError(
                f"Instruments specified as covariates: {instruments_are_covariates}"
            )
        if len(endogenous) > len(instruments):
            raise UnderDeterminedIVError(
                "The IV system is underdetermined. "
                "Please provide as many or more instruments as endogenous variables."
            )
        endogenous_have_multiple_estimation = [
            variable
            for variable in endogenous
            if re.match(_Pattern.multiple_estimation, variable)
        ]
        if endogenous_have_multiple_estimation:
            raise FormulaSyntaxError(
                "Endogenous variables cannot have multiple estimations."
            )
        if len(endogenous) > 1:
            raise FormulaSyntaxError(
                "Multiple endogenous variables are currently not supported."
            )
        return endogenous, instruments


def _parse_multiple_estimation(variables: list[str]) -> _MultipleEstimation:
    single: list[str] = []
    multiple: list[str] = []
    kind: _MultipleEstimationType | None = None
    for variable in variables:
        match = re.match(_Pattern.multiple_estimation, variable)
        if match is None:
            # Single estimation
            single.append(variable)
        elif kind is not None:
            # Multiple "multiple estimation" syntaxes in the formula
            raise DuplicateKeyError(
                "Problem in the RHS of the formula: You cannot use more than one multiple estimation."
            )
        else:
            # Formula term indicates "multiple estimation"
            kind = _MultipleEstimationType[match.group("key")]
            multiple = re.split(_Pattern.args, match.group("variables"))
    return _MultipleEstimation(constant=single, variable=multiple, kind=kind)


def parse(formula: str, intercept: bool = True, sort: bool = False) -> ParsedFormula:
    """
    Parse a fixest model formula.

    Parameters
    ----------
    formula : str
        A one to three sided formula string in the form
        "Y1 + Y2 ~ X1 + X2 | FE1 + FE2 | endogvar ~ exogvar".
    intercept : bool, default=True
        Whether to include an intercept in the model.
    sort : bool, default=False
        Sort variables lexicographically within formula parts.

    Returns
    -------
    ParsedFormula
    """
    # Parse parts of formulas: main part and optional "other" parts (fixed effects and instrumental variables)
    main_part, other_parts = _parse_parts(formula)
    dependent, independent = _parse_dependent_independent(main_part)
    fixed_effects = _parse_fixed_effects(other_parts)
    endogenous, instruments = _parse_instrumental_variable(other_parts, independent)
    if endogenous is not None and instruments is not None:
        if FORMULAIC_FEATURE_FLAG is not DefaultFormulaParser.FeatureFlags.ALL:
            independent = [*endogenous, *independent]
        instruments = ["+".join(instruments)]
    if sort:
        list.sort(independent)
    return ParsedFormula(
        formula=formula,
        dependent=dependent,
        independent=_parse_multiple_estimation(independent),
        fixed_effects=_parse_multiple_estimation(fixed_effects)
        if fixed_effects is not None
        else None,
        endogenous=endogenous,
        instruments=instruments,
        intercept=intercept,
    )
