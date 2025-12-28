import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from pyfixest.errors import (
    DuplicateKeyError,
    EndogVarsAsCovarsError,
    FormulaSyntaxError,
    UnderDeterminedIVError,
)


class _MultipleEstimationType(StrEnum):
    # See https://lrberge.github.io/fixest/reference/stepwise.html
    sw = "sequential stepwise"
    csw = "cumulative stepwise"
    sw0 = "sequential stepwise with zero step"
    csw0 = "cumulative stepwise with zero step"


@dataclass(kw_only=True)
class _MultipleEstimation:
    constant: list[str]
    variable: list[str]
    kind: _MultipleEstimationType = None

    @property
    def is_multiple(self) -> bool:
        return self.kind is not None

    @property
    def steps(self) -> list[str]:
        if not self.is_multiple or self.kind.name.endswith("0"):
            # Add zero step
            estimation_steps = ["+".join(self.constant) if self.constant else "0"]
        else:
            estimation_steps = []
        if self.is_multiple and self.kind.name.startswith("sw"):
            # Sequential stepwise estimation
            estimation_steps.extend(
                ["+".join([*self.constant, v]) for v in self.variable]
            )
        elif self.is_multiple and self.kind.name.startswith("csw"):
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
    dependent: str
    independent: str
    fixed_effects: Optional[str] = None
    endogenous: Optional[str] = None
    instruments: Optional[str] = None

    @property
    def fml(self) -> str:
        formula = f"{self.dependent}~{self.independent}"
        if self.endogenous is not None and self.instruments is not None:
            formula = f"{formula}|{self.endogenous}~{self.instruments}"
        if self.fixed_effects is not None:
            formula = f"{formula}|{self.fixed_effects}"
        return formula

    @property
    def fml_first_stage(self) -> str | None:
        if self.endogenous is not None and self.instruments is not None:
            return f"{self.endogenous}~{self.instruments}+{self.independent}-{self.endogenous}+1"

    @property
    def fml_second_stage(self) -> str:
        return f"{self.dependent}~{self.independent}+1"


@dataclass(kw_only=True, frozen=True)
class _Formulas:
    formula: str
    dependent: list[str]
    independent: _MultipleEstimation
    fixed_effects: _MultipleEstimation = None
    endogenous: list[str] = None
    instruments: list[str] = None

    @property
    def is_multiple(self) -> bool:
        return self.independent.is_multiple or (
            self.fixed_effects is not None and self.fixed_effects.is_multiple
        )

    @property
    def is_fixed_effects(self) -> bool:
        return self.fixed_effects is not None

    @property
    def is_iv(self) -> bool:
        return self.endogenous is not None

    def _collect_formula_kwargs(self) -> dict[str, list[str]]:
        kwargs: dict[str, list[str]] = {
            "dependent": self.dependent,
            "independent": self.independent.steps,
            "fixed_effects": self.fixed_effects.steps if self.is_fixed_effects else "0",
        }
        if self.is_iv:
            kwargs.update(
                {"endogenous": self.endogenous, "instruments": self.instruments}
            )
        return kwargs

    @property
    def FixestFormulaDict(self) -> dict[str, list[Formula]]:
        # Get formulas by group of fixed effects
        estimations = defaultdict(list[Formula])
        dict_of_lists = self._collect_formula_kwargs()
        list_of_kwargs = [
            dict(zip(dict_of_lists.keys(), values))
            for values in itertools.product(*dict_of_lists.values())
        ]
        for kwargs in list_of_kwargs:
            formula = Formula(**kwargs)
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
    parts = re.split(_Pattern.parts, formula.strip())
    if len(parts) > 3:
        raise FormulaSyntaxError(
            f"Formula can have at most 3 parts `dependent ~ independent | fixed effects | endogenous ~ instruments`, "
            f"received {len(parts)}: {formula}"
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
    part_fe: Optional[str] = next((part for part in parts if "~" not in part), None)
    if part_fe is None:
        return None
    else:
        return re.split(_Pattern.variables, part_fe)


def _parse_instrumental_variable(
    parts: list[str],
    independent: list[str],
) -> tuple[list[str] | None, list[str] | None]:
    part_iv: Optional[str] = next((part for part in parts if "~" in part), None)
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
        if len(endogenous) > len(instruments):
            raise UnderDeterminedIVError(
                "The IV system is underdetermined. Please provide as many or more instruments as endogenous variables."
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


def parse(formula: str) -> _Formulas:
    main_part, other_parts = _parse_parts(formula)
    dependent, independent = _parse_dependent_independent(main_part)
    fixed_effects = _parse_fixed_effects(other_parts)
    endogenous, instruments = _parse_instrumental_variable(other_parts, independent)
    if endogenous is not None:
        independent.extend(endogenous)
    return _Formulas(
        formula=formula,
        dependent=dependent,
        independent=_parse_multiple_estimation(independent),
        fixed_effects=_parse_multiple_estimation(fixed_effects),
        endogenous=endogenous,
        instruments=instruments,
    )


if __name__ == "__main__":
    formula: str = "Y ~ 1 | f1 + f2 | X1 ~ Z1"
    new = parse(formula=formula)
    new_lst = new.estimations
