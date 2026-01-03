//! High-level demeaning solver strategies.
//!
//! This module provides the [`Demeaner`] trait for complete demeaning operations,
//! with specialized implementations for different fixed effect counts:
//!
//! - [`SingleFEDemeaner`]: O(n) closed-form solution (1 FE)
//! - [`TwoFEDemeaner`]: Accelerated iteration (2 FEs)
//! - [`MultiFEDemeaner`]: Multi-phase strategy (3+ FEs)

use crate::demean_accelerated::accelerator::{Accelerator, IronsTuckGrand};
use crate::demean_accelerated::projection::{MultiFEProjector, TwoFEProjector};
use crate::demean_accelerated::types::{DemeanContext, FixestConfig};

// =============================================================================
// Demeaner Trait
// =============================================================================

/// A demeaning solver for a specific fixed-effects configuration.
///
/// This trait represents the complete strategy for solving the demeaning
/// problem with a specific number of fixed effects. Implementations handle
/// setup, iteration (if needed), and output reconstruction.
///
/// The trait provides default implementations for scatter/gather operations
/// that transform between observation space and coefficient space.
pub trait Demeaner {
    /// Solve the demeaning problem.
    ///
    /// # Returns
    ///
    /// Tuple of (demeaned_output, iterations_used, converged_flag)
    fn solve(
        ctx: &DemeanContext,
        input: &[f64],
        config: &FixestConfig,
    ) -> (Vec<f64>, usize, bool);

    /// Scatter values from observation space to coefficient space.
    ///
    /// Computes weighted sums of `values` for each group in each FE.
    fn scatter_to_coefficients(ctx: &DemeanContext, values: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; ctx.index.n_coef];
        Self::scatter_inner(ctx, values, None, &mut result);
        result
    }

    /// Scatter residuals from observation space to coefficient space.
    ///
    /// Like [`scatter_to_coefficients`], but first subtracts `baseline` from `values`.
    fn scatter_residuals(ctx: &DemeanContext, values: &[f64], baseline: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; ctx.index.n_coef];
        Self::scatter_inner(ctx, values, Some(baseline), &mut result);
        result
    }

    /// Gather coefficients to observation space and add to output.
    ///
    /// For each observation, looks up its coefficient for each FE and adds to output.
    fn gather_and_add(ctx: &DemeanContext, coef: &[f64], output: &mut [f64]) {
        for q in 0..ctx.index.n_fe {
            let offset = ctx.index.coef_start[q];
            let fe_ids = ctx.index.group_ids_for_fe(q);
            for (i, &g) in fe_ids.iter().enumerate() {
                output[i] += coef[offset + g];
            }
        }
    }

    /// Inner scatter implementation.
    #[inline(always)]
    fn scatter_inner(
        ctx: &DemeanContext,
        values: &[f64],
        baseline: Option<&[f64]>,
        result: &mut [f64],
    ) {
        for q in 0..ctx.index.n_fe {
            let offset = ctx.index.coef_start[q];
            let fe_ids = ctx.index.group_ids_for_fe(q);

            match (ctx.weights.is_uniform, baseline) {
                (true, None) => {
                    for (i, &g) in fe_ids.iter().enumerate() {
                        result[offset + g] += values[i];
                    }
                }
                (true, Some(base)) => {
                    for (i, &g) in fe_ids.iter().enumerate() {
                        result[offset + g] += values[i] - base[i];
                    }
                }
                (false, None) => {
                    for (i, &g) in fe_ids.iter().enumerate() {
                        result[offset + g] += values[i] * ctx.weights.per_obs[i];
                    }
                }
                (false, Some(base)) => {
                    for (i, &g) in fe_ids.iter().enumerate() {
                        result[offset + g] += (values[i] - base[i]) * ctx.weights.per_obs[i];
                    }
                }
            }
        }
    }
}

// =============================================================================
// SingleFEDemeaner
// =============================================================================

/// Demeaner for 1 fixed effect: O(n) closed-form solution.
///
/// No iteration needed - direct computation.
pub struct SingleFEDemeaner;

impl Demeaner for SingleFEDemeaner {
    fn solve(
        ctx: &DemeanContext,
        input: &[f64],
        _config: &FixestConfig,
    ) -> (Vec<f64>, usize, bool) {
        let n_obs = ctx.index.n_obs;
        let mut output = vec![0.0; n_obs];

        // Scatter input to coefficient space (sum of input per group)
        let in_out = Self::scatter_residuals(ctx, input, &output);

        let fe0 = ctx.index.group_ids_for_fe(0);
        let group_weights = ctx.group_weights_for_fe(0);

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

        (output, 0, true)
    }
}

// =============================================================================
// TwoFEDemeaner
// =============================================================================

/// Demeaner for 2 fixed effects: accelerated coefficient-space iteration.
pub struct TwoFEDemeaner;

impl Demeaner for TwoFEDemeaner {
    fn solve(
        ctx: &DemeanContext,
        input: &[f64],
        config: &FixestConfig,
    ) -> (Vec<f64>, usize, bool) {
        let n_obs = ctx.index.n_obs;
        let n0 = ctx.index.n_groups[0];
        let n1 = ctx.index.n_groups[1];
        let n_coef = n0 + n1;

        // Scatter input to coefficient space
        let in_out = Self::scatter_to_coefficients(ctx, input);

        // Initialize coefficient array (unified: [alpha | beta])
        let mut coef = vec![0.0; n_coef];

        // Create buffers and projector
        let mut buffers = IronsTuckGrand::create_buffers(n_coef);
        let mut projector = TwoFEProjector::new(ctx, &in_out, input);

        // Run acceleration loop
        let (iter, converged) =
            IronsTuckGrand::run(&mut projector, &mut coef, &mut buffers, config, config.maxiter);

        // Reconstruct output: input - alpha - beta
        let mut result = vec![0.0; n_obs];
        let fe0 = ctx.index.group_ids_for_fe(0);
        let fe1 = ctx.index.group_ids_for_fe(1);

        for i in 0..n_obs {
            result[i] = input[i] - coef[fe0[i]] - coef[n0 + fe1[i]];
        }

        (result, iter, converged)
    }
}

// =============================================================================
// MultiFEDemeaner
// =============================================================================

/// Demeaner for 3+ fixed effects: multi-phase strategy.
///
/// # Strategy
///
/// 1. **Warmup**: Run all-FE iterations to get initial estimates
/// 2. **2-FE sub-convergence**: Converge on first 2 FEs (faster)
/// 3. **Re-acceleration**: Final all-FE iterations to polish
///
/// # Convergence
///
/// Returns `converged=true` if any phase converges early (before max iterations).
pub struct MultiFEDemeaner;

impl Demeaner for MultiFEDemeaner {
    fn solve(
        ctx: &DemeanContext,
        input: &[f64],
        config: &FixestConfig,
    ) -> (Vec<f64>, usize, bool) {
        let n_obs = ctx.index.n_obs;
        let n_coef = ctx.index.n_coef;
        let n0 = ctx.index.n_groups[0];
        let n1 = ctx.index.n_groups[1];
        let n_coef_2fe = n0 + n1;
        let mut total_iter = 0usize;

        let mut mu = vec![0.0; n_obs];
        let mut coef = vec![0.0; n_coef];

        // Create buffers (one for multi-FE, one for 2-FE sub-convergence)
        let mut multi_buffers = IronsTuckGrand::create_buffers(n_coef);
        let mut two_buffers = IronsTuckGrand::create_buffers(n_coef_2fe);

        // Phase 1: Warmup with all FEs (mu is zeros initially)
        let in_out_phase1 = Self::scatter_to_coefficients(ctx, input);
        let mut projector1 = MultiFEProjector::new(ctx, &in_out_phase1, input);
        let (iter1, converged1) = IronsTuckGrand::run(
            &mut projector1,
            &mut coef,
            &mut multi_buffers,
            config,
            config.iter_warmup,
        );
        total_iter += iter1;
        Self::gather_and_add(ctx, &coef, &mut mu);

        // Determine final convergence status based on which phase completes the algorithm
        let converged = if converged1 {
            // Early convergence in warmup phase
            true
        } else {
            // Phase 2: 2-FE sub-convergence
            let in_out_phase2 = Self::scatter_residuals(ctx, input, &mu);
            let mut coef_2fe = vec![0.0; n_coef_2fe];
            let in_out_2fe: Vec<f64> = in_out_phase2[..n_coef_2fe].to_vec();
            let effective_input: Vec<f64> = (0..n_obs).map(|i| input[i] - mu[i]).collect();

            let mut projector2 = TwoFEProjector::new(ctx, &in_out_2fe, &effective_input);
            let (iter2, converged2) = IronsTuckGrand::run(
                &mut projector2,
                &mut coef_2fe,
                &mut two_buffers,
                config,
                config.maxiter / 2,
            );
            total_iter += iter2;

            // Add 2-FE coefficients to mu
            let fe0 = ctx.index.group_ids_for_fe(0);
            let fe1 = ctx.index.group_ids_for_fe(1);
            for i in 0..n_obs {
                mu[i] += coef_2fe[fe0[i]] + coef_2fe[n0 + fe1[i]];
            }

            // Phase 3: Re-acceleration with all FEs (unless 2-FE converged fully)
            let remaining = config.maxiter.saturating_sub(total_iter);
            if remaining > 0 {
                let in_out_phase3 = Self::scatter_residuals(ctx, input, &mu);
                coef.fill(0.0);
                let mut projector3 = MultiFEProjector::new(ctx, &in_out_phase3, input);
                let (iter3, converged3) = IronsTuckGrand::run(
                    &mut projector3,
                    &mut coef,
                    &mut multi_buffers,
                    config,
                    remaining,
                );
                total_iter += iter3;
                Self::gather_and_add(ctx, &coef, &mut mu);
                converged3
            } else {
                // No remaining iterations, use phase 2 convergence status
                converged2
            }
        };

        // Compute output: input - mu
        let mut output = vec![0.0; n_obs];
        for i in 0..n_obs {
            output[i] = input[i] - mu[i];
        }

        (output, total_iter, converged)
    }
}

// =============================================================================
// Entry Point
// =============================================================================

/// Demean a single variable using the appropriate solver.
///
/// Dispatches to the appropriate [`Demeaner`] implementation based on FE count.
///
/// # Panics
///
/// Panics in debug builds if `input.len() != ctx.index.n_obs`.
pub fn demean_single(
    ctx: &DemeanContext,
    input: &[f64],
    config: &FixestConfig,
) -> (Vec<f64>, usize, bool) {
    debug_assert_eq!(
        input.len(),
        ctx.index.n_obs,
        "input length ({}) must match number of observations ({})",
        input.len(),
        ctx.index.n_obs
    );

    match ctx.index.n_fe {
        1 => SingleFEDemeaner::solve(ctx, input, config),
        2 => TwoFEDemeaner::solve(ctx, input, config),
        _ => MultiFEDemeaner::solve(ctx, input, config),
    }
}
