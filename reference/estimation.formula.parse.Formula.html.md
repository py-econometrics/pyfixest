# estimation.formula.parse.Formula { #pyfixest.estimation.formula.parse.Formula }

```python
estimation.formula.parse.Formula(
    _second_stage,
    _fixed_effects=None,
    _first_stage=None,
)
```

A formulaic-compliant formula.

## Attributes

| Name | Description |
| --- | --- |
| [endogenous](#pyfixest.estimation.formula.parse.Formula.endogenous) | Endogenous variables of an instrumental variable specification. |
| [exogenous](#pyfixest.estimation.formula.parse.Formula.exogenous) | Exogenous aka covariates aka independent variables. |
| [first_stage](#pyfixest.estimation.formula.parse.Formula.first_stage) | The first stage formula of an instrumental variable specification. |
| [fixed_effects](#pyfixest.estimation.formula.parse.Formula.fixed_effects) | The fixed effects of a formula. |
| [formula](#pyfixest.estimation.formula.parse.Formula.formula) | Full fixest-style formula. |
| [second_stage](#pyfixest.estimation.formula.parse.Formula.second_stage) | The second stage formula. |

## Methods

| Name | Description |
| --- | --- |
| [parse](#pyfixest.estimation.formula.parse.Formula.parse) | Parse fixest-style formula. In case of multiple estimation syntax, |
| [parse_to_dict](#pyfixest.estimation.formula.parse.Formula.parse_to_dict) | Group parsed formulas into dictionary keyed by fixed effects. |

### parse { #pyfixest.estimation.formula.parse.Formula.parse }

```python
estimation.formula.parse.Formula.parse(formula)
```

Parse fixest-style formula. In case of multiple estimation syntax,
returns a list of multiple regression formulas.

### parse_to_dict { #pyfixest.estimation.formula.parse.Formula.parse_to_dict }

```python
estimation.formula.parse.Formula.parse_to_dict(formula)
```

Group parsed formulas into dictionary keyed by fixed effects.