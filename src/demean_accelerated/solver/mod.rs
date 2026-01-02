//! Solver implementations for different FE counts.
//!
//! Dispatches to specialized solvers based on number of fixed effects:
//! - 1 FE: O(n) closed-form solution
//! - 2 FE: Specialized accelerated iteration
//! - 3+ FE: Multi-phase strategy with 2-FE sub-convergence

mod single_fe;
mod two_fe;
mod multi_fe;

use single_fe::solve_single_fe;

use crate::demean_accelerated::types::{FEInfo, FixestConfig};

/// Demean a single variable using coefficient-space iteration.
///
/// Dispatches to the appropriate solver based on FE count.
pub fn demean_single(
    fe_info: &FEInfo,
    input: &[f64],
    config: &FixestConfig,
) -> (Vec<f64>, usize, bool) {
    match fe_info.structure.n_fe {
        1 => {
            let output = solve_single_fe(fe_info, input);
            (output, 0, true)
        }
        2 => two_fe::solve_two_fe(fe_info, input, config),
        _ => multi_fe::solve_multi_fe(fe_info, input, config),
    }
}
