# Formula

``` python
Formula(_second_stage, _fixed_effects=None, _first_stage=None)
```

A formulaic-compliant formula.

Splits a fixest-style formula into second stage, fixed effects and, for IV models, first stage. Use `parse()` instead of calling the class directly. `parse()` also expands the multiple estimation operators (`sw`, `sw0`, `csw`, `csw0`, `mvsw`) into one `Formula` per model. This is an internal API. Formulas are written as strings and passed to `feols()`. See the [formula syntax tutorial](../tutorials/formula-syntax.llms.md).

## Examples

``` python
from pyfixest.estimation.formula.parse import Formula

fml = Formula.parse("Y ~ X1 + X2 | f1 + f2")[0]
fml.second_stage, fml.fixed_effects
```

    ('Y ~ X1 + X2', 'f1 + f2')

Stepwise syntax expands into one formula per estimated model.

``` python
Formula.parse("Y ~ X1 + csw(X2, X3)")
```

    [Formula(_second_stage='Y ~ X1 + X2', _fixed_effects=None, _first_stage=None),
     Formula(_second_stage='Y ~ X1 + X2 + X3', _fixed_effects=None, _first_stage=None)]

## Attributes

| Name | Description |
|----|----|
| [Formula.endogenous](#pyfixest.estimation.formula.parse.Formula.endogenous) | Endogenous variables of an instrumental variable specification. |
| [Formula.exogenous](#pyfixest.estimation.formula.parse.Formula.exogenous) | Exogenous aka covariates aka independent variables. |
| [Formula.first_stage](#pyfixest.estimation.formula.parse.Formula.first_stage) | The first stage formula of an instrumental variable specification. |
| [Formula.fixed_effects](#pyfixest.estimation.formula.parse.Formula.fixed_effects) | The fixed effects of a formula. |
| [Formula.formula](#pyfixest.estimation.formula.parse.Formula.formula) | Full fixest-style formula. |
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
