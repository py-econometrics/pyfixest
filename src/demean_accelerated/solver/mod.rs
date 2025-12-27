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

use crate::demean_accelerated::types::{
    FEInfo, FixestConfig, AccelerationStrategy, ConvergenceCriterion,
    IronsTuck, FixestConvergence,
};

/// Demean a single variable using coefficient-space iteration.
///
/// Dispatches to the appropriate solver based on FE count.
/// Uses default strategies (IronsTuck, FixestConvergence).
pub fn demean_single(
    fe_info: &FEInfo,
    input: &[f64],
    config: &FixestConfig,
) -> (Vec<f64>, usize, bool) {
    demean_single_with_strategies(fe_info, input, config, &IronsTuck, &FixestConvergence)
}

/// Demean a single variable with custom strategies.
///
/// Allows plugging in different acceleration and convergence strategies.
pub fn demean_single_with_strategies<A, C>(
    fe_info: &FEInfo,
    input: &[f64],
    config: &FixestConfig,
    acceleration: &A,
    convergence: &C,
) -> (Vec<f64>, usize, bool)
where
    A: AccelerationStrategy,
    C: ConvergenceCriterion,
{
    match fe_info.n_fe {
        1 => {
            let output = solve_single_fe(fe_info, input);
            (output, 0, true)
        }
        2 => two_fe::solve_two_fe(fe_info, input, config, acceleration, convergence),
        _ => multi_fe::solve_multi_fe(fe_info, input, config, acceleration, convergence),
    }
}
