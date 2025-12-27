//! Single FE solver: O(n) closed-form solution.

use crate::demean_accelerated::types::FEInfo;

/// Solve single-FE demeaning in closed form.
///
/// # Algorithm
/// For each group g: coef[g] = sum(input[i] for i in group g) / count(g)
/// Output: output[i] = input[i] - coef[fe[i]]
///
/// No iteration needed - direct O(n) computation.
pub fn solve_single_fe(fe_info: &FEInfo, input: &[f64]) -> Vec<f64> {
    let n_obs = fe_info.n_obs;
    let mut output = vec![0.0; n_obs];

    // Compute in_out (sum of input per group)
    let in_out = fe_info.compute_in_out(input, &output);

    let fe0 = fe_info.fe_ids_slice(0);
    let sw0 = fe_info.sum_weights_slice(0);

    // coef[g] = in_out[g] / sw[g]
    let coef: Vec<f64> = in_out
        .iter()
        .zip(sw0.iter())
        .map(|(&io, &sw)| io / sw)
        .collect();

    // output[i] = input[i] - coef[fe0[i]]
    for i in 0..n_obs {
        output[i] = input[i] - coef[fe0[i]];
    }

    output
}
