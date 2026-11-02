"""OLS and IV specifications parsed and materialized by formula."""

from __future__ import annotations

import re
from dataclasses import dataclass

import formula

from pyfixest.errors import FormulaSyntaxError


def split_top_level(source: str, separator: str) -> list[str]:
    """Split formula structure without splitting quoted names or nested calls."""
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    quote = ""
    escaped = False
    for index, char in enumerate(source):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\" and quote != "`":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in "\"'`":
            quote = char
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or {"(": ")", "[": "]", "{": "}"}[stack.pop()] != char:
                raise FormulaSyntaxError("Unbalanced brackets in formula.")
        elif char == separator and not stack:
            parts.append(source[start:index].strip())
            start = index + 1
    if quote or stack:
        raise FormulaSyntaxError("Unterminated quote or bracket in formula.")
    parts.append(source[start:].strip())
    return parts


def _native_intercept(source: str) -> str:
    """Translate the conventional zero intercept without changing call arguments."""
    result: list[str] = []
    calls: list[bool] = []
    quote = ""
    for index, char in enumerate(source):
        if quote:
            result.append(char)
            if char == quote and (index == 0 or source[index - 1] != "\\"):
                quote = ""
            continue
        if char in "`'\"":
            quote = char
        elif char == "(":
            before = source[:index].rstrip()
            calls.append(bool(before and (before[-1].isalnum() or before[-1] == "_")))
        elif char == ")":
            calls.pop()
        if (
            char == "0"
            and not any(calls)
            and (index == 0 or source[index - 1] in " (+-*/:")
            and (index + 1 == len(source) or source[index + 1] in " )+-*/:")
        ):
            result.append("-1")
        else:
            result.append(char)
    return "".join(result)


def categorical_calls(source: str) -> list[str]:
    """Find C() calls, ignoring quoted column names and string arguments."""
    tokens = re.finditer(
        r"""`(?:``|[^`])*`|'(?:\\.|[^'])*'|"(?:\\.|[^"])*"|\bC\s*\(|[()]""", source
    )
    start = None
    depth = 0
    calls = []
    for token in tokens:
        value = token.group()
        if value.startswith("C"):
            if start is None:
                start = token.start()
            depth += 1
        elif start is not None and value == "(":
            depth += 1
        elif start is not None and value == ")":
            depth -= 1
            if depth == 0:
                calls.append(source[start : token.end()])
                start = None
    return calls


@dataclass(frozen=True, slots=True)
class LinearFormula:
    """Separate IV equations while leaving expression parsing to Rust."""

    dependent: str
    exogenous: str
    fixed_effects_key: str | None = None
    endogenous: str | None = None
    instruments: str | None = None

    @property
    def is_instrumental_variable(self) -> bool:
        """Whether a first-stage equation was supplied."""
        return self.endogenous is not None

    @property
    def is_fixed_effects(self) -> bool:
        """Whether a fixed-effect expression was supplied."""
        return self.fixed_effects_key is not None

    @property
    def has_varying_slopes(self) -> bool:
        """Varying slopes are rejected on the native linear path."""
        return False

    @property
    def absorbs_intercept(self) -> bool:
        """Ordinary fixed effects absorb the model intercept."""
        return self.is_fixed_effects

    @property
    def second_stage(self) -> str:
        """Render the structural equation for native materialization."""
        rhs = self.exogenous
        if self.endogenous is not None:
            rhs += f" + ({self.endogenous})"
        result = f"{self.dependent} ~ {rhs}"
        if self.fixed_effects_key is not None:
            result += f" | {self.fixed_effects_key}"
        return result

    def render(self) -> str:
        """Render the user-facing model label including its IV block."""
        rhs = self.exogenous
        if self.endogenous is not None:
            rhs += f" + [{self.endogenous} ~ {self.instruments}]"
        result = f"{self.dependent} ~ {rhs}"
        if self.fixed_effects_key is not None:
            result += f" | {self.fixed_effects_key}"
        return result

    def as_first_stage(self) -> LinearFormula:
        """Regress the endogenous variable on included and excluded instruments."""
        if self.endogenous is None or self.instruments is None:
            raise TypeError("Not an instrumental variable specification.")
        return LinearFormula(
            dependent=self.endogenous,
            exogenous=f"(({self.exogenous}) - ({self.endogenous})) + ({self.instruments})",
            fixed_effects_key=self.fixed_effects_key,
        )

    @classmethod
    def parse(cls, source: str) -> list[LinearFormula]:
        """Expand pyfixest's multiple-estimation syntax before native parsing."""
        from pyfixest.estimation.formula.parse import _expand_all_multiple_estimation

        pipe_parts = split_top_level(source, "|")
        sides = split_top_level(pipe_parts[0], "~")
        if len(sides) == 2:
            responses = split_top_level(sides[0], "+")
            if len(responses) > 1:
                pipe_parts[0] = f"sw({', '.join(responses)}) ~ {sides[1]}"
                source = " | ".join(pipe_parts)
        return [
            cls._parse_one(expanded)
            for expanded in _expand_all_multiple_estimation(source)
        ]

    @classmethod
    def _parse_one(cls, source: str) -> LinearFormula:
        """Accept ordinary and bracketed/pipe IV formulas; reject unsupported syntax."""
        parts = split_top_level(source, "|")
        for part in parts[1:]:
            unquoted_part = re.sub(
                r'''`(?:``|[^`])*`|'(?:\\.|[^'])*'|"(?:\\.|[^"])*"''', "", part
            )
            if "[" in unquoted_part and "~" in unquoted_part:
                raise FormulaSyntaxError(
                    "Bracketed IV syntax belongs before the fixed-effects pipe."
                )
        if len(parts) > 2 and len(split_top_level(parts[1], "~")) > 1:
            raise FormulaSyntaxError("The IV equation must be the final pipe part.")
        if len(parts) > 3:
            raise FormulaSyntaxError("A formula may contain at most three pipe parts.")
        if len(parts) == 3 and len(split_top_level(parts[-1], "~")) != 2:
            raise FormulaSyntaxError("The IV equation must be the final pipe part.")
        main = split_top_level(parts[0], "~")
        if len(main) != 2 or not all(main):
            raise FormulaSyntaxError(
                "Specify a response and regressors separated by '~'."
            )
        dependent, exogenous = main
        fixed_effects = None
        endogenous = instruments = None
        if len(parts) > 1:
            if len(split_top_level(parts[-1], "~")) > 1:
                iv = split_top_level(parts.pop(), "~")
                if len(iv) != 2 or not all(iv):
                    raise FormulaSyntaxError(
                        "Specify one endogenous expression and instruments."
                    )
                endogenous, instruments = iv
            if len(parts) == 2 and parts[1] not in ("0", "1"):
                fixed_effects = parts[1]
        terms = split_top_level(exogenous, "+")
        blocks = [term for term in terms if term.startswith("[") and term.endswith("]")]
        if blocks:
            if len(blocks) != 1 or endogenous is not None:
                raise FormulaSyntaxError(
                    "Only one instrumental-variable block is supported."
                )
            iv = split_top_level(blocks[0][1:-1], "~")
            if len(iv) != 2 or not all(iv):
                raise FormulaSyntaxError(
                    "Use an IV block of the form [endogenous ~ instruments]."
                )
            endogenous, instruments = iv
            exogenous = " + ".join(term for term in terms if term != blocks[0]) or "1"
        unquoted = re.sub(
            r'''`(?:``|[^`])*`|'(?:\\.|[^'])*'|"(?:\\.|[^"])*"''', "", source
        )
        if fixed_effects and "[" in re.sub(r"`(?:``|[^`])*`", "", fixed_effects):
            raise NotImplementedError(
                "Varying slopes are not supported by the formula engine yet."
            )
        if re.search(r"\bi\s*\(", unquoted):
            raise NotImplementedError(
                "i() is not supported by the formula engine yet; use C() for categorical terms."
            )
        result = cls(
            dependent,
            _native_intercept(exogenous),
            fixed_effects,
            endogenous,
            _native_intercept(instruments) if instruments is not None else None,
        )
        try:
            formula.parse(result.second_stage)
            if result.is_instrumental_variable:
                formula.parse(result.as_first_stage().second_stage)
        except (SyntaxError, ValueError) as error:
            raise FormulaSyntaxError(str(error)) from error
        return result
