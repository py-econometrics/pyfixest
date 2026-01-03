//! High-level demeaning solver strategies.
//!
//! This module provides the [`Demeaner`] trait for complete demeaning operations,
//! with specialized implementations for different fixed effect counts:
//!
//! - [`SingleFEDemeaner`]: O(n) closed-form solution (1 FE)
//! - [`TwoFEDemeaner`]: Accelerated iteration (2 FEs)
//! - [`MultiFEDemeaner`]: Multi-phase strategy (3+ FEs)
//!
//! # Buffer Reuse
//!
//! Demeaners own their working buffers, allowing reuse across multiple `solve()` calls.
//! This is important for parallel processing where each thread can have its own
//! demeaner instance that reuses buffers across columns.

use crate::demean_accelerated::accelerator::{Accelerator, IronsTuckGrand, IronsTuckGrandBuffers};
use crate::demean_accelerated::projection::{MultiFEProjector, TwoFEProjector};
use crate::demean_accelerated::types::{DemeanContext, FixestConfig};

// =============================================================================
// Demeaner Trait
// =============================================================================

/// A demeaning solver for a specific fixed-effects configuration.
///
/// Demeaners own references to their context and configuration, as well as
/// working buffers that are reused across multiple `solve()` calls.
pub trait Demeaner {
    /// Solve the demeaning problem.
    ///
    /// # Returns
    ///
    /// Tuple of (demeaned_output, iterations_used, converged_flag)
    fn solve(&mut self, input: &[f64]) -> (Vec<f64>, usize, bool);
}

// =============================================================================
// SingleFEDemeaner
// =============================================================================

/// Demeaner for 1 fixed effect: O(n) closed-form solution.
///
/// No iteration or buffers needed - direct computation.
pub struct SingleFEDemeaner<'a> {
    ctx: &'a DemeanContext,
}

impl<'a> SingleFEDemeaner<'a> {
    /// Create a new single-FE demeaner.
    #[inline]
    pub fn new(ctx: &'a DemeanContext) -> Self {
        Self { ctx }
    }
}

impl Demeaner for SingleFEDemeaner<'_> {
    fn solve(&mut self, input: &[f64]) -> (Vec<f64>, usize, bool) {
        let n_obs = self.ctx.index.n_obs;
        let output = vec![0.0; n_obs];

        // Scatter input to coefficient space (sum of input per group)
        let in_out = self.ctx.scatter_residuals(input, &output);

        let fe0 = self.ctx.index.group_ids_for_fe(0);
        let group_weights = self.ctx.group_weights_for_fe(0);

        // coef[g] = in_out[g] / group_weights[g]
        let coef: Vec<f64> = in_out
            .iter()
            .zip(group_weights.iter())
            .map(|(&io, &sw)| io / sw)
            .collect();

        // output[i] = input[i] - coef[fe0[i]]
        let output: Vec<f64> = (0..n_obs).map(|i| input[i] - coef[fe0[i]]).collect();

        (output, 0, true)
    }
}

// =============================================================================
// TwoFEDemeaner
// =============================================================================

/// Demeaner for 2 fixed effects: accelerated coefficient-space iteration.
///
/// Owns working buffers that are reused across multiple `solve()` calls.
pub struct TwoFEDemeaner<'a> {
    ctx: &'a DemeanContext,
    config: &'a FixestConfig,
    /// Coefficient array [alpha | beta], reused across solves
    coef: Vec<f64>,
    /// Acceleration buffers, reused across solves
    buffers: IronsTuckGrandBuffers,
}

impl<'a> TwoFEDemeaner<'a> {
    /// Create a new two-FE demeaner with pre-allocated buffers.
    #[inline]
    pub fn new(ctx: &'a DemeanContext, config: &'a FixestConfig) -> Self {
        let n0 = ctx.index.n_groups[0];
        let n1 = ctx.index.n_groups[1];
        let n_coef = n0 + n1;

        Self {
            ctx,
            config,
            coef: vec![0.0; n_coef],
            buffers: IronsTuckGrand::create_buffers(n_coef),
        }
    }
}

impl Demeaner for TwoFEDemeaner<'_> {
    fn solve(&mut self, input: &[f64]) -> (Vec<f64>, usize, bool) {
        let n_obs = self.ctx.index.n_obs;
        let n0 = self.ctx.index.n_groups[0];

        // Scatter input to coefficient space
        let in_out = self.ctx.scatter_to_coefficients(input);

        // Reset coefficient array for this solve
        self.coef.fill(0.0);

        // Create projector (lightweight, references in_out and input)
        let mut projector = TwoFEProjector::new(self.ctx, &in_out, input);

        // Run acceleration loop with reused buffers
        let (iter, converged) = IronsTuckGrand::run(
            &mut projector,
            &mut self.coef,
            &mut self.buffers,
            self.config,
            self.config.maxiter,
        );

        // Reconstruct output: input - alpha - beta
        let fe0 = self.ctx.index.group_ids_for_fe(0);
        let fe1 = self.ctx.index.group_ids_for_fe(1);

        let result: Vec<f64> = (0..n_obs)
            .map(|i| input[i] - self.coef[fe0[i]] - self.coef[n0 + fe1[i]])
            .collect();

        (result, iter, converged)
    }
}

// =============================================================================
// MultiFEDemeaner
// =============================================================================

/// Working buffers for multi-FE demeaning.
///
/// Groups the observation-space and coefficient-space arrays that are
/// reused across multiple `solve()` calls.
struct MultiFEBuffers {
    /// Accumulated fixed effects per observation (observation-space)
    mu: Vec<f64>,
    /// Coefficient array for all FEs (coefficient-space)
    coef: Vec<f64>,
    /// Coefficient array for 2-FE sub-convergence (coefficient-space, first 2 FEs only)
    coef_2fe: Vec<f64>,
    /// Effective input after subtracting mu (observation-space)
    effective_input: Vec<f64>,
}

impl MultiFEBuffers {
    /// Create new buffers with the given dimensions.
    fn new(n_obs: usize, n_coef: usize, n_coef_2fe: usize) -> Self {
        Self {
            mu: vec![0.0; n_obs],
            coef: vec![0.0; n_coef],
            coef_2fe: vec![0.0; n_coef_2fe],
            effective_input: vec![0.0; n_obs],
        }
    }

    /// Reset all buffers to zero for a new solve.
    #[inline]
    fn reset(&mut self) {
        self.mu.fill(0.0);
        self.coef.fill(0.0);
    }
}

/// Demeaner for 3+ fixed effects: multi-phase strategy.
///
/// Owns working buffers that are reused across multiple `solve()` calls.
///
/// # Strategy
///
/// 1. **Warmup**: Run all-FE iterations to get initial estimates
/// 2. **2-FE sub-convergence**: Converge on first 2 FEs (faster)
/// 3. **Re-acceleration**: Final all-FE iterations to polish
pub struct MultiFEDemeaner<'a> {
    ctx: &'a DemeanContext,
    config: &'a FixestConfig,
    /// Working buffers for coefficient and observation arrays
    buffers: MultiFEBuffers,
    /// Acceleration buffers for multi-FE iterations
    multi_acc: IronsTuckGrandBuffers,
    /// Acceleration buffers for 2-FE sub-convergence
    two_acc: IronsTuckGrandBuffers,
}

impl<'a> MultiFEDemeaner<'a> {
    /// Create a new multi-FE demeaner with pre-allocated buffers.
    #[inline]
    pub fn new(ctx: &'a DemeanContext, config: &'a FixestConfig) -> Self {
        let n_obs = ctx.index.n_obs;
        let n_coef = ctx.index.n_coef;
        let n0 = ctx.index.n_groups[0];
        let n1 = ctx.index.n_groups[1];
        let n_coef_2fe = n0 + n1;

        Self {
            ctx,
            config,
            buffers: MultiFEBuffers::new(n_obs, n_coef, n_coef_2fe),
            multi_acc: IronsTuckGrand::create_buffers(n_coef),
            two_acc: IronsTuckGrand::create_buffers(n_coef_2fe),
        }
    }
}

impl Demeaner for MultiFEDemeaner<'_> {
    fn solve(&mut self, input: &[f64]) -> (Vec<f64>, usize, bool) {
        let n_obs = self.ctx.index.n_obs;
        let n0 = self.ctx.index.n_groups[0];
        let n1 = self.ctx.index.n_groups[1];
        let n_coef_2fe = n0 + n1;
        let mut total_iter = 0usize;

        // Reset buffers for this solve
        self.buffers.reset();

        // Phase 1: Warmup with all FEs (mu is zeros initially)
        let in_out_phase1 = self.ctx.scatter_to_coefficients(input);
        let mut projector1 = MultiFEProjector::new(self.ctx, &in_out_phase1, input);
        let (iter1, converged1) = IronsTuckGrand::run(
            &mut projector1,
            &mut self.buffers.coef,
            &mut self.multi_acc,
            self.config,
            self.config.iter_warmup,
        );
        total_iter += iter1;
        self.ctx.gather_and_add(&self.buffers.coef, &mut self.buffers.mu);

        // Determine final convergence status based on which phase completes the algorithm
        let converged = if converged1 {
            // Early convergence in warmup phase
            true
        } else {
            // Phase 2: 2-FE sub-convergence
            let in_out_phase2 = self.ctx.scatter_residuals(input, &self.buffers.mu);
            self.buffers.coef_2fe.fill(0.0);
            let in_out_2fe: Vec<f64> = in_out_phase2[..n_coef_2fe].to_vec();

            // Compute effective input: input - mu
            for i in 0..n_obs {
                self.buffers.effective_input[i] = input[i] - self.buffers.mu[i];
            }

            let mut projector2 =
                TwoFEProjector::new(self.ctx, &in_out_2fe, &self.buffers.effective_input);
            let (iter2, converged2) = IronsTuckGrand::run(
                &mut projector2,
                &mut self.buffers.coef_2fe,
                &mut self.two_acc,
                self.config,
                self.config.maxiter / 2,
            );
            total_iter += iter2;

            // Add 2-FE coefficients to mu
            let fe0 = self.ctx.index.group_ids_for_fe(0);
            let fe1 = self.ctx.index.group_ids_for_fe(1);
            for i in 0..n_obs {
                self.buffers.mu[i] +=
                    self.buffers.coef_2fe[fe0[i]] + self.buffers.coef_2fe[n0 + fe1[i]];
            }

            // Phase 3: Re-acceleration with all FEs (unless 2-FE converged fully)
            let remaining = self.config.maxiter.saturating_sub(total_iter);
            if remaining > 0 {
                let in_out_phase3 = self.ctx.scatter_residuals(input, &self.buffers.mu);
                self.buffers.coef.fill(0.0);
                let mut projector3 = MultiFEProjector::new(self.ctx, &in_out_phase3, input);
                let (iter3, converged3) = IronsTuckGrand::run(
                    &mut projector3,
                    &mut self.buffers.coef,
                    &mut self.multi_acc,
                    self.config,
                    remaining,
                );
                total_iter += iter3;
                self.ctx.gather_and_add(&self.buffers.coef, &mut self.buffers.mu);
                converged3
            } else {
                // No remaining iterations, use phase 2 convergence status
                converged2
            }
        };

        // Compute output: input - mu
        let output: Vec<f64> = (0..n_obs).map(|i| input[i] - self.buffers.mu[i]).collect();

        (output, total_iter, converged)
    }
}
