# Keep solver state consistent across backends

Numerical backends must publish arrays that describe the same fit. After
step-halving in IWLS, the predictor and mean describe the shorter step while
the coefficients and weighted least-squares objects still describe the full
update. Require another full IWLS iteration before accepting convergence.
The probit deviance calculation suppresses intermediate log warnings
and sums the finite contributions.

SciPy LSQR takes one right-hand side at a time. Stopping codes 3 and 6 denote
a condition-number limit and 7 denotes the iteration limit. Those codes
do not certify accuracy. Codes 0, 1, 2, 4, and 5 report an exact zero
solution or residual-based success at requested or machine precision.
See the [SciPy LSQR stopping table](https://github.com/scipy/scipy/blob/v1.17.1/scipy/sparse/linalg/_isolve/lsqr.py#L334-L341).

## Backend representation

- Once an adaptive demeaning handle has built an additive preconditioner,
  later solves reuse it without restarting the adaptive search. Torch LSMR
  always uses its diagonal preconditioner. The option resolver emits the
  existing warning for incompatible requests.
- Torch LSMR derives the diagonal scaling from group weight totals and guards
  zero-weight groups. Build the preconditioned operator once across columns.
  Use CPU NumPy storage without a copy when possible, coalesce sparse COO for
  multiplication, and select batched sparse multiplication at the benchmarked
  crossover for each device.
- The compiled Torch scalar step packs state and convergence constants into
  tensors. This lets one fused Metal or CUDA kernel fit within Metal's buffer
  limit. Keep the input slots stable, pass through the scalar state the caller
  updates, and append the coefficients for the vector update and convergence
  estimates. The same zeta value feeds both the next scalar step and the
  current vector update. The norm of the solution comes from the actual vector.
  Use an overflow-safe hypotenuse where Metal lacks `torch.hypot` code generation
  and guard zero denominators. Like SciPy, start updating the running minimum
  for condition estimation on the second iteration.
- The singleton kernel uses column-major unsigned 32-bit codes. Transposing
  an allocation with reversed axes gives the required layout without a
  copy. Gelbach decomposition uses CSR for row stacking and CSC for selecting
  columns.
