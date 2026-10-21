"""THROWAWAY: can prepared i() names replace caller-frame inspection?

Run from an existing environment:
    pixi run --as-is -e py312 python <worktree>/pyfixest/estimation/formula/prototype_explicit_i_names.py

Related: #1763, #1738, PR #1948. This module changes only its own process.

Observed: 21 standard estimation scenarios plus a custom i() override match
the PR, including predictions, formula roundtrips and retained-formula refits.
The prototype removes frame inspection, but still parses an AST and adds
private arguments to materialized factor expressions. Preparation must happen
after evaluation context resolution so custom i() overrides stay untouched.
"""

from __future__ import annotations

import ast
import importlib
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import formulaic
import numpy as np
from formulaic import FactorValues
from formulaic.parser.types import Factor, Term
from formulaic.utils.code import sanitize_variable_names

import pyfixest as pf
from pyfixest.estimation.formula import FORMULAIC_TRANSFORMS

interaction = importlib.import_module(
    "pyfixest.estimation.formula.transforms.factor_interaction"
)
matrices = importlib.import_module("pyfixest.estimation.formula.model_matrix")


def argument_text(node, aliases):
    """The same quoting rule as the PR, without looking at a caller frame."""
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in aliases:
            child.id = f"`{aliases[child.id]}`"
    return ast.unparse(node)


def prepare_factor(factor):
    if factor.eval_method is not Factor.EvalMethod.PYTHON or "i(" not in factor.expr:
        return factor
    aliases = {}
    code = ast.parse(sanitize_variable_names(factor.expr, {}, aliases), mode="eval")
    changed = False
    for node in ast.walk(code):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "i"
        ):
            continue
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            continue
        args = dict(zip(("data", "var2"), node.args, strict=False))
        args.update((kw.arg, kw.value) for kw in node.keywords if kw.arg in ("data", "var2"))
        if "data" not in args:
            continue
        # Parse copies: restoring backticks for labels must not mutate the
        # expression that Formulaic will evaluate.
        names = tuple(
            argument_text(ast.parse(ast.unparse(args[key]), mode="eval").body, aliases)
            if key in args else None
            for key in ("data", "var2")
        )
        node.keywords.append(ast.keyword(arg="_names", value=ast.Constant(value=names)))
        changed = True
    if not changed:
        return factor
    for node in ast.walk(code):
        if isinstance(node, ast.Name) and node.id in aliases:
            node.id = f"`{aliases[node.id]}`"
    return Factor(
        ast.unparse(code), eval_method=factor.eval_method, kind=factor.kind,
        metadata=factor.metadata, token=factor.token,
    )


def prepare_leaf(formula):
    return formulaic.formula.SimpleFormula(
        Term([prepare_factor(factor) for factor in term.factors], origin=term.origin)
        for term in formula
    )


def explicit_i(data, var2=None, *, ref=None, ref2=None, bin=None, bin2=None, _names=None):
    """Prototype replacement for i(); source names arrive as an argument."""
    factor_name = _names[0] if _names else interaction._get_series_name(data, default="factor")
    var2_name = None if var2 is None else (
        (_names[1] if _names else None) or interaction._get_series_name(var2, default="var")
    )

    def encoder(values, **encoding):
        return interaction._encode_i(
            values=values, factor_name=factor_name, var2_name=var2_name,
            ref=ref, ref2=ref2, bin=bin, bin2=bin2, **encoding,
        )

    return FactorValues(
        {"__data__": data, "__var2__": var2} if var2 is not None else data,
        kind="categorical", spans_intercept=var2 is None, encoder=encoder,
    )


def install_prototype():
    def prepared_formula(formula):
        return formula._map(prepare_leaf, as_type=type(formula))

    def frame_lookup_forbidden(**kwargs):
        raise AssertionError("Prototype called the PR's frame lookup")

    matrices._prototype_prepare_i_names = prepared_formula
    FORMULAIC_TRANSFORMS["i"] = explicit_i
    interaction._argument_expressions = frame_lookup_forbidden


def snapshot(case, data):
    label, estimator, formula, options = case
    result = getattr(pf, estimator)(formula, data=data, **options)
    models = result.all_fitted_models if hasattr(result, "all_fitted_models") else {result.model.formula: result}
    saved = {}
    for name, fit in models.items():
        assert "_names=" not in name
        assert "_names=" not in fit.model.formula
        assert all("_names=" not in term for term in fit.coef().index)
        prediction = None
        if not options.get("lean") and estimator == "feols" and "[" not in formula:
            prediction = fit.predict(newdata=data.iloc[:8])
        saved[name] = (fit.coef(), fit.se(), fit.sample_info.n_obs, prediction)
    return saved, result


def main():
    warnings.simplefilter("ignore")
    data = pf.get_data().dropna().reset_index(drop=True)
    data["my f"] = data["f1"]
    data["country"] = np.where(data.f1 == 1, "Cote d'Ivoire", "other")
    data["sample"] = np.where(np.arange(len(data)) % 2, "odd", "even")
    rng = np.random.default_rng(1738)
    data["weight"] = rng.integers(1, 4, size=len(data))
    data["count"] = rng.poisson(2, size=len(data))
    data["binary"] = rng.binomial(1, 0.5, size=len(data))
    cases = [
        ("colliding slopes", "feols", "Y ~ i(f3, X2) + i(f3, np.square(X2))", {}),
        ("unnamed log expressions", "feols", "Y ~ i(f3, log(X2 + 10)) + i(f3, log(Z2 + 10))", {}),
        ("quoted column", "feols", "Y ~ i(`my f`, np.square(X2), ref=1)", {}),
        ("quoted expression", "feols", "Y ~ i(`my f` + 1, np.square(X2))", {}),
        ("apostrophe in reference", "feols", 'Y ~ i(country, X2, ref="Cote d\'Ivoire")', {}),
        ("positional None", "feols", "Y ~ i(f1, None)", {}),
        ("keyword None", "feols", "Y ~ i(data=f1, var2=None)", {}),
        ("unpacked slope", "feols", "Y ~ i(f1, **dict(var2=X2))", {}),
        ("starred arguments", "feols", "Y ~ i(*(f1, X2))", {}),
        ("unpacked options", "feols", "Y ~ i(f1, np.square(X2), **dict(ref=1))", {}),
        ("interaction term", "feols", "Y ~ i(f1, X2):X1", {}),
        ("fixed effects", "feols", "Y ~ i(f3, np.exp(X2 / 5)) | f1", {}),
        ("aweights", "feols", "Y ~ i(f3, np.square(X2)) | f1", {"weights": "weight"}),
        ("fweights", "feols", "Y ~ i(f3, np.square(X2)) | f1", {"weights": "weight", "weights_type": "fweights"}),
        ("multiple estimation", "feols", "Y ~ sw(i(f3, X2), i(f3, np.exp(X2 / 5))) | f1", {}),
        ("reversed model order", "feols", "Y ~ sw(i(f3, np.exp(X2 / 5)), i(f3, X2)) | f1", {}),
        ("split samples", "feols", "Y ~ i(f3, np.square(X2))", {"fsplit": "sample"}),
        ("IV stages", "feols", "Y ~ i(f3, np.square(X2)) + [X1 ~ Z1] | f1", {}),
        ("stripped data", "feols", "Y ~ i(f3, np.square(X2)) | f1", {"lean": True, "store_data": False}),
        ("Poisson", "fepois", "count ~ i(f3, np.square(X2)) | f1", {}),
        ("logit", "feglm", "binary ~ i(f3, np.square(X2)) | f1", {"family": "logit"}),
    ]
    print("THROWAWAY prototype: explicit prepared names versus PR #1948")
    baseline = [snapshot(case, data)[0] for case in cases]
    custom_context = {"i": lambda values: 2 * values}
    custom_before = pf.feols("Y ~ i(X1)", data=data, context=custom_context)
    install_prototype()
    for case, before in zip(cases, baseline, strict=True):
        after, fit = snapshot(case, data)
        assert list(before) == list(after), case[0]
        for name in before:
            old_coef, old_se, old_nobs, old_pred = before[name]
            new_coef, new_se, new_nobs, new_pred = after[name]
            assert old_coef.index.equals(new_coef.index), case[0]
            assert old_nobs == new_nobs, case[0]
            np.testing.assert_allclose(new_coef, old_coef, rtol=1e-10, atol=1e-10, err_msg=case[0] + " coefficients")
            np.testing.assert_allclose(new_se, old_se, rtol=1e-10, atol=1e-10, err_msg=case[0] + " standard errors")
            if old_pred is not None:
                np.testing.assert_allclose(new_pred, old_pred, rtol=1e-10, atol=1e-10, equal_nan=True, err_msg=case[0] + " predictions")
        print(f"PASS {case[0]}: names, coefficients, SEs, observations, available predictions")
    example = pf.feols("Y ~ i(`my f`, np.square(X2), ref=1)", data=data)
    public = example.model.formula
    internal = str(example.model.model_spec["second_stage"].rhs.formula)
    print("PUBLIC FORMULA:", public)
    print("INTERNAL SPEC:", internal)
    print("COEFFICIENTS:", example.coef().index[:3].tolist())
    roundtrip = pf.feols(public, data=data)
    np.testing.assert_allclose(roundtrip.coef(), example.coef(), rtol=1e-10, atol=1e-10, err_msg="formula roundtrip")
    from pyfixest.estimation.refit import refit
    from pyfixest.estimation.internals.model_state import VcovSpec
    rebuilt = refit(fit=example, data=data, vcov=VcovSpec.from_user_input("iid"))
    np.testing.assert_allclose(rebuilt.coef(), example.coef(), rtol=1e-10, atol=1e-10, err_msg="retained-formula refit")
    print("PASS formula roundtrip and retained-formula refit")
    print("CUSTOM i() BEFORE:", custom_before.coef().index.tolist())
    custom_after = pf.feols("Y ~ i(X1)", data=data, context=custom_context)
    np.testing.assert_allclose(custom_after.coef(), custom_before.coef(), rtol=1e-10, atol=1e-10, err_msg="custom i override")
    assert custom_before.coef().index.equals(custom_after.coef().index)
    print("CUSTOM i() AFTER:", custom_after.coef().index.tolist(), "PASS")
    print("VERDICT: frame lookup eliminated; AST parsing remains, internal expressions gain _names.")


if __name__ == "__main__":
    main()
