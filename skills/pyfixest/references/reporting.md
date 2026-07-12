# Reporting and result APIs

Use fit.tidy(), fit.coef(), fit.se(), fit.tstat(), fit.pvalue(), and
fit.confint() for machine-readable results. Use fit.summary(), pf.etable,
pf.dtable, pf.coefplot, pf.iplot, and pf.qplot for presentation.

The reporting wrappers are explicit methods on single results and FixestMulti:
summary(), etable(), coefplot(), and iplot(). Pass type=df, md, tex, or another
documented output type to etable as needed.

For IV, call first_stage() before reading first_stage_model or
first_stage_f_statistic; call IV_Diag() before effective_f_statistic. Do not
rely on private underscore attributes.

Read pyfixest/docs/pages/tutorials/regression-tables.md and the installed Feols
or Feiv reference page for supported post-estimation methods.
