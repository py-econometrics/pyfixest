# Preserve reporting and simulation semantics

## Reporting

Reporting uses pyfixest's significance codes even when the table dependency
has a different default. An unsplit model reports the full sample and no split
variable. Plot labels distinguish models with identical formulas. Coefficient
labels handle both explicit categorical terms and columns originally supplied
as categoricals. Parsing a numeric label as a float before converting to an
integer lets a template display `2.0` as `2`. Interaction plots select the terms
produced by `i()`, and the data passed to plotting backends preserves their
expected layout.

In decomposition plots, red indicates movement away from zero and green
indicates movement toward it. Display the red components first. Annotation
order keeps the outer value close to the bar while preserving the color
associated with each line. When locating a public warning, count the call to
`warnings.warn` as level one and skip import machinery frames, as Python does.

## Simulation

The children-and-earnings and immigration-and-wages simulations distinguish
the causal effect, the confounder's effects on treatment and outcome, and
instrument strength. Those separate parameters let examples compare the
known causal coefficient with omitted-variable bias. The adoption simulation
sets the take-up probabilities for encouraged and unencouraged groups to
identify compliers and always-takers. Its reference first stage is 0.55, its
intent-to-treat effect is 1.10, and its complier effect is 2.0.

The staggered panel simulations construct untreated potential outcomes with
unit noise and a stronger weekend effect, then apply heterogeneous treatment
effects. The cohort paths include a decaying bump, a logarithmic first-week
effect, and a sinusoid. The twins design uses shared ability within pairs and
correlates education with ability, with a true log-wage return of 0.08. The
worker-firm panel switches roughly one fifth of workers per year after the
initial assignment. The motherhood panel leaves some units never treated,
builds in slower earnings growth before birth, and varies the post-birth
effect by region.
