# Formula syntax

Use y ~ x1 + x2 for OLS, y ~ x1 | firm + year for fixed effects, and
y ~ x1 | firm + year | endogenous ~ instrument for IV. The IV part is
endogenous ~ excluded_instruments; use the three-part form when fixed effects
are present.

Use i(category, ref=...) for indicator expansion or i(category, variable) for
interactions. Use firm ^ year to create an interacted fixed effect.

For multiple estimation, use sw, sw0, csw, csw0, or mvsw; combine them
deliberately because they fan out to several models. Use split for one fit per
group and fsplit to include the full sample too.

Read pyfixest/docs/pages/tutorials/formula-syntax.md before translating a
complex R fixest formula.
