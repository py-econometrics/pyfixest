# estimation.formula.factor_interaction.factor_interaction { #pyfixest.estimation.formula.factor_interaction.factor_interaction }

```python
estimation.formula.factor_interaction.factor_interaction(
    data,
    var2=None,
    *,
    ref=None,
    ref2=None,
    bin=None,
    bin2=None,
)
```

Fixest-style i() operator for categorical encoding with interactions.

Args:
    data: The categorical variable
    var2: Optional second variable for interaction (continuous or categorical)
    ref: Reference level to drop from data
    ref2: Reference level to drop from var2 (if categorical)
    bin: Dict mapping new_level -> [old_levels] for binning

Naming convention (matches R fixest):
    i(cyl)           -> cyl::4, cyl::6, cyl::8
    i(cyl, ref=4)    -> cyl::6, cyl::8
    i(cyl, wt)       -> cyl::4:wt, cyl::6:wt, cyl::8:wt
    i(cyl, wt, ref=4) -> cyl::6:wt, cyl::8:wt