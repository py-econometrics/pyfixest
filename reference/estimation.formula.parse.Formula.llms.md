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

    ('Y ~ X1 + X2', f1 + f2)

Stepwise syntax expands into one formula per estimated model.

``` python
Formula.parse("Y ~ X1 + csw(X2, X3)")
```

    [Y ~ 1 + X1 + X2, Y ~ 1 + X1 + X2 + X3]

## Attributes

| Name | Description |
|----|----|
| [Formula.dependent](#pyfixest.estimation.formula.parse.Formula.dependent) | The dependent variable. |
| [Formula.endogenous](#pyfixest.estimation.formula.parse.Formula.endogenous) | Endogenous variables of an instrumental variable specification. |
| [Formula.exogenous](#pyfixest.estimation.formula.parse.Formula.exogenous) | Exogenous aka covariates aka independent variables. |
| [Formula.first_stage](#pyfixest.estimation.formula.parse.Formula.first_stage) | The first stage formula of an instrumental variable specification. |
| [Formula.fixed_effects](#pyfixest.estimation.formula.parse.Formula.fixed_effects) | The fixed effects of a formula. |
| [Formula.fixed_effects_wrapped](#pyfixest.estimation.formula.parse.Formula.fixed_effects_wrapped) | Wrapped fixed effects for proper encoding. |
| [Formula.formula](#pyfixest.estimation.formula.parse.Formula.formula) | The string representation of the formula. |
| [Formula.instruments](#pyfixest.estimation.formula.parse.Formula.instruments) | Instruments of an instrumental variable specification. |
| [Formula.is_fixed_effects](#pyfixest.estimation.formula.parse.Formula.is_fixed_effects) | Boolean indicating whether the formula is a fixed effects specification. |
| [Formula.is_instrumental_variable](#pyfixest.estimation.formula.parse.Formula.is_instrumental_variable) | Boolean indicating whether the formula is an instrumental variable specification. |
| [Formula.second_stage](#pyfixest.estimation.formula.parse.Formula.second_stage) | The second stage formula. |

## Methods

| Name | Description |
|----|----|
| [Formula.parse](#pyfixest.estimation.formula.parse.Formula.parse) | Parse fixest-style formula. In case of multiple estimation syntax, |
| [Formula.parse_to_dict](#pyfixest.estimation.formula.parse.Formula.parse_to_dict) | Group parsed formulas into dictionary keyed by fixed effects. |

### Formula.parse

``` python
parse(formula)
```

Parse fixest-style formula. In case of multiple estimation syntax, returns a list of multiple regression formulas.

### Formula.parse_to_dict

``` python
parse_to_dict(formula)
```

Group parsed formulas into dictionary keyed by fixed effects.
