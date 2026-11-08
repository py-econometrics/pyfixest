# Formula

``` python
Formula(_formula)
```

A formulaic-compliant formula.

Splits a fixest-style formula into second stage, fixed effects and, for IV models, first stage. Use `parse()` instead of calling the class directly. `parse()` also expands the multiple estimation operators (`sw`, `sw0`, `csw`, `csw0`, `mvsw`) into one `Formula` per model. This is an internal API. Formulas are written as strings and passed to `feols()`. See the [formula syntax tutorial](../tutorials/formula-syntax.llms.md).

## Examples

``` python
from pyfixest.estimation.formula.parse import Formula

fml = Formula.parse("Y ~ X1 + X2 | f1 + f2")[0]
fml.second_stage, fml.fixed_effects
```

    (.lhs:
         Y
     .rhs:
         1 + X1 + X2,
     f1 + f2)

Stepwise syntax expands into one formula per estimated model.

``` python
Formula.parse("Y ~ X1 + csw(X2, X3)")
```

    [Y ~ 1 + X1 + X2, Y ~ 1 + X1 + X2 + X3]

## Attributes

| Name | Description |
|----|----|
| [Formula.absorbs_intercept](#pyfixest.estimation.formula.parse.Formula.absorbs_intercept) | Whether any fixed effect supplies group-specific intercepts. |
| [Formula.dependent](#pyfixest.estimation.formula.parse.Formula.dependent) | The dependent variable. |
| [Formula.endogenous](#pyfixest.estimation.formula.parse.Formula.endogenous) | Endogenous variables of an instrumental variable specification. |
| [Formula.exogenous](#pyfixest.estimation.formula.parse.Formula.exogenous) | Exogenous aka covariates aka independent variables. |
| [Formula.fixed_effect_slopes](#pyfixest.estimation.formula.parse.Formula.fixed_effect_slopes) | Distinct varying-slope terms in normalized specification order. |
| [Formula.fixed_effect_specifications](#pyfixest.estimation.formula.parse.Formula.fixed_effect_specifications) | Fixed effects represented as symbolic `within::Effect` terms. |
| [Formula.fixed_effects](#pyfixest.estimation.formula.parse.Formula.fixed_effects) | The fixed effects of a formula. |
| [Formula.fixed_effects_key](#pyfixest.estimation.formula.parse.Formula.fixed_effects_key) | Rendered fixed effects, grouping models that share demeaning caches. |
| [Formula.formulaic_fixed_effects](#pyfixest.estimation.formula.parse.Formula.formulaic_fixed_effects) | Fixed-effect level terms in Formulaic’s internal representation. |
| [Formula.has_varying_slopes](#pyfixest.estimation.formula.parse.Formula.has_varying_slopes) | Whether any fixed effect has varying slopes, as in `f1[z]`. |
| [Formula.instruments](#pyfixest.estimation.formula.parse.Formula.instruments) | Instruments of an instrumental variable specification. |
| [Formula.is_fixed_effects](#pyfixest.estimation.formula.parse.Formula.is_fixed_effects) | Boolean indicating whether the formula is a fixed effects specification. |
| [Formula.is_instrumental_variable](#pyfixest.estimation.formula.parse.Formula.is_instrumental_variable) | Boolean indicating whether the formula is an instrumental variable specification. |
| [Formula.second_stage](#pyfixest.estimation.formula.parse.Formula.second_stage) | Parsed response and regressors for second-stage materialization. |

## Methods

| Name | Description |
|----|----|
| [Formula.as_first_stage](#pyfixest.estimation.formula.parse.Formula.as_first_stage) | Return the parsed OLS specification for an IV first stage. |
| [Formula.parse](#pyfixest.estimation.formula.parse.Formula.parse) | Parse fixest-style formula. In case of multiple estimation syntax, |
| [Formula.render](#pyfixest.estimation.formula.parse.Formula.render) | Render an expanded formula preserving lookup and expression semantics. |

### Formula.as_first_stage

``` python
as_first_stage()
```

Return the parsed OLS specification for an IV first stage.

### Formula.parse

``` python
parse(formula)
```

Parse fixest-style formula. In case of multiple estimation syntax, returns a list of multiple regression formulas.

### Formula.render

``` python
render()
```

Render an expanded formula preserving lookup and expression semantics.

This is the readable public model name. Distinct parsed formulas render to distinct strings, so the rendered text also identifies models and fixed-effect groups internally. Materialization uses the retained parsed stages.
