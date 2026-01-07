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

use crate::demean::accelerator::IronsTuckGrand;
use crate::demean::projection::{MultiFEProjector, TwoFEProjector};
use crate::demean::types::{ConvergenceState, DemeanContext, DemeanResult, FixestConfig};

// =============================================================================
// Demeaner Trait
// =============================================================================

/// A demeaning solver for a specific fixed-effects configuration.
///
/// Demeaners own references to their context and configuration, as well as
/// working buffers that are reused across multiple `solve()` calls.
pub trait Demeaner {
    /// Solve the demeaning problem for a single column.
    ///
    /// # Returns
    ///
    /// A `DemeanResult` containing:
    /// - `demeaned`: The input with fixed effects removed
    /// - `success`: Whether the algorithm converged
    /// - `iterations`: Number of iterations (0 for closed-form solutions)
    fn solve(&mut self, input: &[f64]) -> DemeanResult;
}

// =============================================================================
// SingleFEDemeaner
// =============================================================================

/// Demeaner for 1 fixed effect: O(n) closed-form solution.
///
/// Owns a reusable buffer for the coefficient-space sums.
pub struct SingleFEDemeaner<'a> {
    ctx: &'a DemeanContext,
    /// Weighted sums per group (Dᵀ · input), reused across solves.
    coef_sums_buffer: Vec<f64>,
}

impl<'a> SingleFEDemeaner<'a> {
    /// Create a new single-FE demeaner.
    #[inline]
    pub fn new(ctx: &'a DemeanContext) -> Self {
        Self {
            ctx,
            coef_sums_buffer: vec![0.0; ctx.index.n_coef],
        }
    }
}

impl Demeaner for SingleFEDemeaner<'_> {
    fn solve(&mut self, input: &[f64]) -> DemeanResult {
        let n_obs = self.ctx.index.n_obs;

        // Apply Dᵀ to get coefficient-space sums (reuses buffer)
        self.ctx.apply_design_matrix_t(input, &mut self.coef_sums_buffer);

        let fe0 = self.ctx.index.group_ids_for_fe(0);
        let group_weights = self.ctx.group_weights_for_fe(0);

        // output[i] = input[i] - group_mean[fe0[i]]
        // where group_mean[g] = coef_sums_buffer[g] / group_weights[g]
        let demeaned: Vec<f64> = (0..n_obs)
            .map(|i| input[i] - self.coef_sums_buffer[fe0[i]] / group_weights[fe0[i]])
            .collect();

        // Single FE is a closed-form solution, always converges in 0 iterations
        DemeanResult {
            demeaned,
            convergence: ConvergenceState::Converged,
            iterations: 0,
        }
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
    /// Weighted sums per group (Dᵀ · input), reused across solves.
    coef_sums_buffer: Vec<f64>,
    /// Coefficient array [alpha | beta], reused across calls to solve.
    coef: Vec<f64>,
    /// Accelerator with internal buffers, reused across solves
    accelerator: IronsTuckGrand,
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
            coef_sums_buffer: vec![0.0; n_coef],
            coef: vec![0.0; n_coef],
            accelerator: IronsTuckGrand::new(*config, n_coef),
        }
    }
}

impl Demeaner for TwoFEDemeaner<'_> {
    fn solve(&mut self, input: &[f64]) -> DemeanResult {
        let n_obs = self.ctx.index.n_obs;
        let n0 = self.ctx.index.n_groups[0];

        // Apply Dᵀ to get coefficient-space sums (reuses buffer)
        self.ctx.apply_design_matrix_t(input, &mut self.coef_sums_buffer);

        // Reset coefficient array for this call to solve
        self.coef.fill(0.0);

        // Create the projector (lightweight, references coef_sums_buffer and input)
        let mut projector = TwoFEProjector::new(self.ctx, &self.coef_sums_buffer, input);

        // Run acceleration loop
        let (iter, convergence) = self
            .accelerator
            .run(&mut projector, &mut self.coef, self.config.maxiter);

        // Reconstruct output: input - alpha - beta
        let fe0 = self.ctx.index.group_ids_for_fe(0);
        let fe1 = self.ctx.index.group_ids_for_fe(1);

        let demeaned: Vec<f64> = (0..n_obs)
            .map(|i| input[i] - self.coef[fe0[i]] - self.coef[n0 + fe1[i]])
            .collect();

        DemeanResult {
            demeaned,
            convergence,
            iterations: iter,
        }
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
    /// Working coefficient array for accelerator (reset each phase)
    coef: Vec<f64>,
    /// Coefficient array for 2-FE sub-convergence (coefficient-space, first 2 FEs only)
    coef_2fe: Vec<f64>,
    /// Effective input after subtracting mu (observation-space).
    effective_input: Vec<f64>,
    /// Weighted sums per group (Dᵀ · input), reused across phases.
    coef_sums_buffer: Vec<f64>,
}

impl MultiFEBuffers {
    /// Create new buffers with the given dimensions.
    fn new(n_obs: usize, n_coef: usize, n_coef_2fe: usize) -> Self {
        Self {
            mu: vec![0.0; n_obs],
            coef: vec![0.0; n_coef],
            coef_2fe: vec![0.0; n_coef_2fe],
            effective_input: vec![0.0; n_obs],
            coef_sums_buffer: vec![0.0; n_coef],
        }
    }

    /// Reset all buffers to zero for a new call to solve.
    #[inline]
    fn reset(&mut self) {
        self.mu.fill(0.0);
        self.coef.fill(0.0);
    }
}

/// Demeaner for 3+ fixed effects: multiphase strategy.
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
    /// Accelerator for multi-FE iterations
    multi_acc: IronsTuckGrand,
    /// Accelerator for 2-FE sub-convergence
    two_acc: IronsTuckGrand,
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
            multi_acc: IronsTuckGrand::new(*config, n_coef),
            two_acc: IronsTuckGrand::new(*config, n_coef_2fe),
        }
    }

    /// Phase 1: Warmup with all FEs to get initial estimates.
    fn warmup_phase(&mut self, input: &[f64]) -> (usize, ConvergenceState) {
        self.ctx
            .apply_design_matrix_t(input, &mut self.buffers.coef_sums_buffer);
        let mut projector = MultiFEProjector::new(self.ctx, &self.buffers.coef_sums_buffer, input);

        let (iter, convergence) = self
            .multi_acc
            .run(&mut projector, &mut self.buffers.coef, self.config.iter_warmup);

        self.ctx
            .apply_design_matrix(&self.buffers.coef, &mut self.buffers.mu);
        (iter, convergence)
    }

    /// Phase 2: Fast 2-FE sub-convergence on the first two fixed effects.
    fn two_fe_convergence_phase(&mut self, input: &[f64]) -> (usize, ConvergenceState) {
        let n_obs = self.ctx.index.n_obs;
        let n0 = self.ctx.index.n_groups[0];
        let n1 = self.ctx.index.n_groups[1];
        let n_coef_2fe = n0 + n1;

        // Compute residuals: input - mu
        for i in 0..n_obs {
            self.buffers.effective_input[i] = input[i] - self.buffers.mu[i];
        }

        // Apply Dᵀ to residuals (reuses buffer, only first 2 FEs used below)
        self.ctx
            .apply_design_matrix_t(&self.buffers.effective_input, &mut self.buffers.coef_sums_buffer);

        // Run 2-FE acceleration (use slice of coef_sums_buffer, no copy needed)
        self.buffers.coef_2fe.fill(0.0);
        let mut projector = TwoFEProjector::new(
            self.ctx,
            &self.buffers.coef_sums_buffer[..n_coef_2fe],
            &self.buffers.effective_input,
        );
        let (iter, convergence) = self.two_acc.run(
            &mut projector,
            &mut self.buffers.coef_2fe,
            self.config.maxiter / 2,
        );

        // Add 2-FE coefficients to mu
        self.add_2fe_coefficients_to_mu();
        (iter, convergence)
    }

    /// Phase 3: Final re-acceleration with all FEs.
    fn reacceleration_phase(
        &mut self,
        input: &[f64],
        used_iter: usize,
    ) -> (usize, ConvergenceState) {
        let remaining = self.config.maxiter.saturating_sub(used_iter);
        if remaining == 0 {
            return (0, ConvergenceState::NotConverged);
        }

        // Compute residuals: input - mu
        for i in 0..self.ctx.index.n_obs {
            self.buffers.effective_input[i] = input[i] - self.buffers.mu[i];
        }

        self.ctx
            .apply_design_matrix_t(&self.buffers.effective_input, &mut self.buffers.coef_sums_buffer);
        self.buffers.coef.fill(0.0);

        let mut projector = MultiFEProjector::new(self.ctx, &self.buffers.coef_sums_buffer, input);
        let (iter, convergence) =
            self.multi_acc
                .run(&mut projector, &mut self.buffers.coef, remaining);

        self.ctx
            .apply_design_matrix(&self.buffers.coef, &mut self.buffers.mu);
        (iter, convergence)
    }

    /// Add 2-FE coefficients to the accumulated mu buffer.
    fn add_2fe_coefficients_to_mu(&mut self) {
        let n0 = self.ctx.index.n_groups[0];
        let fe0 = self.ctx.index.group_ids_for_fe(0);
        let fe1 = self.ctx.index.group_ids_for_fe(1);

        for i in 0..self.ctx.index.n_obs {
            self.buffers.mu[i] +=
                self.buffers.coef_2fe[fe0[i]] + self.buffers.coef_2fe[n0 + fe1[i]];
        }
    }

    /// Compute the final output and return the result.
    fn finalize_output(
        &self,
        input: &[f64],
        iter: usize,
        convergence: ConvergenceState,
    ) -> DemeanResult {
        let demeaned: Vec<f64> = input
            .iter()
            .zip(self.buffers.mu.iter())
            .map(|(&x, &mu)| x - mu)
            .collect();

        DemeanResult {
            demeaned,
            convergence,
            iterations: iter,
        }
    }
}

impl Demeaner for MultiFEDemeaner<'_> {
    fn solve(&mut self, input: &[f64]) -> DemeanResult {
        self.buffers.reset();

        // Phase 1: Warmup with all FEs
        let (iter1, conv1) = self.warmup_phase(input);
        if conv1 == ConvergenceState::Converged {
            return self.finalize_output(input, iter1, conv1);
        }

        // Phase 2: 2-FE sub-convergence (refines only first 2 FEs)
        // Note: Don't return early on Phase 2 convergence!
        // Phase 2 only refines the first 2 FEs. The 3rd+ FEs still need Phase 3.
        let (iter2, _conv2) = self.two_fe_convergence_phase(input);
        let total_iter = iter1 + iter2;

        // Phase 3: Re-acceleration with all FEs
        let (iter3, conv3) = self.reacceleration_phase(input, total_iter);

        self.finalize_output(input, total_iter + iter3, conv3)
    }
}
