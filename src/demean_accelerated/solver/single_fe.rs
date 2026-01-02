//! Single FE solver: O(n) closed-form solution.

use crate::demean_accelerated::types::DemeanContext;

/// Solve single-FE demeaning in closed form.
///
/// # Algorithm
/// For each group g: coef[g] = sum(input[i] for i in group g) / count(g)
/// Output: output[i] = input[i] - coef[fe[i]]
///
/// No iteration needed - direct O(n) computation.
pub fn solve_single_fe(ctx: &DemeanContext, input: &[f64]) -> Vec<f64> {
    let n_obs = ctx.index.n_obs;
    let mut output = vec![0.0; n_obs];

    // Scatter input to coefficient space (sum of input per group)
    let in_out = ctx.scatter_residuals_to_coefficients(input, &output);

    let fe0 = ctx.index.group_ids_for_fe(0);
    let group_weights = ctx.weights.group_weights_for_fe(0, &ctx.index);

    // coef[g] = in_out[g] / group_weights[g]
    let coef: Vec<f64> = in_out
        .iter()
        .zip(group_weights.iter())
        .map(|(&io, &sw)| io / sw)
        .collect();

    // output[i] = input[i] - coef[fe0[i]]
    for i in 0..n_obs {
        output[i] = input[i] - coef[fe0[i]];
    }

    output
}
