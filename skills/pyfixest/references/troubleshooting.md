# Troubleshooting

Start with the received value in a PyFixest error. Formula errors usually need a
valid y ~ rhs | fixed_effects | endogenous ~ instruments form. Vcov errors
usually need a supported name, existing cluster columns, or HAC identifiers.

MissingStoredDataError means a post-estimation operation needs data removed by
store_data=False or lean=True. Refit with store_data=True; for eligible vcov()
calls, pass the original data explicitly.

Unsupported combinations are intentional: check fixed effects, IV, weights,
multiple estimation, and the model family before retrying. Install optional
dependencies only when selecting their paths, for example numba, lets-plot, or
torch.

Read pyfixest/docs/pages/troubleshooting.md first, then the formula, inference,
or demeaner reference named by the error.
