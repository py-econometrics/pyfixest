//! Coefficient-space demeaning matching fixest's algorithm exactly.
//!
//! This is a direct port of fixest's demeaning.cpp, using coefficient-space
//! iteration rather than residual-space iteration.

/// Pre-computed FE information for coefficient-space iteration.
pub struct FEInfo {
    pub n_obs: usize,
    pub n_fe: usize,
    /// Group IDs for each FE: fe_ids[q][i] = group ID for observation i in FE q
    pub fe_ids: Vec<Vec<usize>>,
    /// Number of groups per FE
    pub n_groups: Vec<usize>,
    /// Starting index of each FE's coefficients
    pub coef_start: Vec<usize>,
    /// Total number of coefficients
    pub n_coef_total: usize,
    /// Sum of weights per group: sum_weights[q][g]
    pub sum_weights: Vec<Vec<f64>>,
    /// Sample weights
    pub weights: Vec<f64>,
    /// Whether all weights are 1.0 (optimization)
    pub is_unweighted: bool,
}

impl FEInfo {
    pub fn new(
        n_obs: usize,
        n_fe: usize,
        group_ids: &[usize], // flat [n_obs * n_fe], row-major
        n_groups: &[usize],
        weights: &[f64],
    ) -> Self {
        // Check if unweighted
        let is_unweighted = weights.iter().all(|&w| (w - 1.0).abs() < 1e-10);

        // Extract per-FE group IDs
        let mut fe_ids = vec![vec![0usize; n_obs]; n_fe];
        for i in 0..n_obs {
            for q in 0..n_fe {
                fe_ids[q][i] = group_ids[i * n_fe + q];
            }
        }

        // Coefficient starting indices
        let mut coef_start = vec![0usize; n_fe];
        for q in 1..n_fe {
            coef_start[q] = coef_start[q - 1] + n_groups[q - 1];
        }
        let n_coef_total: usize = n_groups.iter().sum();

        // Sum of weights per group
        let mut sum_weights = Vec::with_capacity(n_fe);
        for q in 0..n_fe {
            let mut sw = vec![0.0; n_groups[q]];
            for i in 0..n_obs {
                sw[fe_ids[q][i]] += weights[i];
            }
            // Avoid division by zero
            for s in &mut sw {
                if *s == 0.0 {
                    *s = 1.0;
                }
            }
            sum_weights.push(sw);
        }

        Self {
            n_obs,
            n_fe,
            fe_ids,
            n_groups: n_groups.to_vec(),
            coef_start,
            n_coef_total,
            sum_weights,
            weights: weights.to_vec(),
            is_unweighted,
        }
    }

    /// Compute sum of weighted (input - output) for each coefficient.
    /// This is computed ONCE at the start and never changes.
    pub fn compute_in_out(&self, input: &[f64], output: &[f64]) -> Vec<f64> {
        let mut in_out = vec![0.0; self.n_coef_total];

        if self.is_unweighted {
            for q in 0..self.n_fe {
                let start = self.coef_start[q];
                let fe_q = &self.fe_ids[q];
                for i in 0..self.n_obs {
                    in_out[start + fe_q[i]] += input[i] - output[i];
                }
            }
        } else {
            for q in 0..self.n_fe {
                let start = self.coef_start[q];
                let fe_q = &self.fe_ids[q];
                for i in 0..self.n_obs {
                    in_out[start + fe_q[i]] += (input[i] - output[i]) * self.weights[i];
                }
            }
        }

        in_out
    }

    /// Compute output from coefficients: output[i] = input[i] - sum_q(coef[fe_q[i]])
    pub fn compute_output(&self, coef: &[f64], input: &[f64], output: &mut [f64]) {
        output.copy_from_slice(input);
        for q in 0..self.n_fe {
            let start = self.coef_start[q];
            let fe_q = &self.fe_ids[q];
            for i in 0..self.n_obs {
                output[i] -= coef[start + fe_q[i]];
            }
        }
    }
}

/// Fixest's continue_crit: returns true if should CONTINUE (not converged).
#[inline]
fn continue_crit(a: f64, b: f64, diff_max: f64) -> bool {
    let diff = (a - b).abs();
    (diff > diff_max) && (diff / (0.1 + a.abs()) > diff_max)
}

/// Check if should continue on coefficient slice.
fn should_continue(x: &[f64], gx: &[f64], tol: f64) -> bool {
    for i in 0..x.len() {
        if continue_crit(x[i], gx[i], tol) {
            return true;
        }
    }
    false
}

/// Fixest's stopping_crit for SSR.
#[inline]
fn stopping_crit(a: f64, b: f64, diff_max: f64) -> bool {
    let diff = (a - b).abs();
    (diff < diff_max) || (diff / (0.1 + a.abs()) < diff_max)
}

/// Irons-Tuck acceleration: X = GGX - coef * (GGX - GX)
#[inline(always)]
fn irons_tuck_update(x: &mut [f64], gx: &[f64], ggx: &[f64]) -> bool {
    let n = x.len();
    let mut vprod = 0.0;
    let mut ssq = 0.0;

    // SAFETY: x, gx, ggx all have the same length n
    for i in 0..n {
        unsafe {
            let gx_i = *gx.get_unchecked(i);
            let ggx_i = *ggx.get_unchecked(i);
            let x_i = *x.get_unchecked(i);
            let delta_gx = ggx_i - gx_i;
            let delta2_x = delta_gx - gx_i + x_i;
            vprod += delta_gx * delta2_x;
            ssq += delta2_x * delta2_x;
        }
    }

    if ssq == 0.0 {
        return true;
    }

    let coef = vprod / ssq;
    for i in 0..n {
        unsafe {
            let gx_i = *gx.get_unchecked(i);
            let ggx_i = *ggx.get_unchecked(i);
            *x.get_unchecked_mut(i) = ggx_i - coef * (ggx_i - gx_i);
        }
    }

    false
}

/// Configuration matching fixest defaults.
#[derive(Clone, Copy)]
pub struct FixestConfig {
    pub tol: f64,
    pub maxiter: usize,
    pub iter_warmup: usize,
    pub iter_proj_after_acc: usize,
    pub iter_grand_acc: usize,
}

impl Default for FixestConfig {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            maxiter: 100_000,
            iter_warmup: 15,
            iter_proj_after_acc: 40,
            iter_grand_acc: 4,
        }
    }
}

// =============================================================================
// 2-FE Coefficient-Space Implementation (matching compute_fe_coef_2)
// =============================================================================

/// 2-FE projection: Given alpha coefficients, compute new alpha via beta.
/// This matches fixest's compute_fe_coef_2 which avoids N-length intermediates.
#[inline(always)]
fn project_2fe(
    fe_info: &FEInfo,
    in_out: &[f64],
    alpha_in: &[f64],
    alpha_out: &mut [f64],
    beta: &mut [f64],
) {
    let n0 = fe_info.n_groups[0];
    let n1 = fe_info.n_groups[1];
    let n_obs = fe_info.n_obs;
    let fe0 = &fe_info.fe_ids[0];
    let fe1 = &fe_info.fe_ids[1];
    let sw0 = &fe_info.sum_weights[0];
    let sw1 = &fe_info.sum_weights[1];
    let weights = &fe_info.weights;

    // Step 1: Compute beta from alpha_in
    // beta[g] = (in_out[n0+g] - sum_{i:fe1[i]=g} alpha[fe0[i]] * w[i]) / sw1[g]
    beta[..n1].copy_from_slice(&in_out[n0..n0 + n1]);

    // SAFETY: fe0[i] < n0 (alpha_in.len()), fe1[i] < n1 (beta.len()) by construction
    if fe_info.is_unweighted {
        for i in 0..n_obs {
            unsafe {
                let g1 = *fe1.get_unchecked(i);
                let g0 = *fe0.get_unchecked(i);
                *beta.get_unchecked_mut(g1) -= *alpha_in.get_unchecked(g0);
            }
        }
    } else {
        for i in 0..n_obs {
            unsafe {
                let g1 = *fe1.get_unchecked(i);
                let g0 = *fe0.get_unchecked(i);
                *beta.get_unchecked_mut(g1) -= *alpha_in.get_unchecked(g0) * *weights.get_unchecked(i);
            }
        }
    }

    for g in 0..n1 {
        unsafe { *beta.get_unchecked_mut(g) /= *sw1.get_unchecked(g) };
    }

    // Step 2: Compute alpha_out from beta
    // alpha[g] = (in_out[g] - sum_{i:fe0[i]=g} beta[fe1[i]] * w[i]) / sw0[g]
    alpha_out[..n0].copy_from_slice(&in_out[..n0]);

    // SAFETY: fe0[i] < n0 (alpha_out.len()), fe1[i] < n1 (beta.len()) by construction
    if fe_info.is_unweighted {
        for i in 0..n_obs {
            unsafe {
                let g0 = *fe0.get_unchecked(i);
                let g1 = *fe1.get_unchecked(i);
                *alpha_out.get_unchecked_mut(g0) -= *beta.get_unchecked(g1);
            }
        }
    } else {
        for i in 0..n_obs {
            unsafe {
                let g0 = *fe0.get_unchecked(i);
                let g1 = *fe1.get_unchecked(i);
                *alpha_out.get_unchecked_mut(g0) -= *beta.get_unchecked(g1) * *weights.get_unchecked(i);
            }
        }
    }

    for g in 0..n0 {
        unsafe { *alpha_out.get_unchecked_mut(g) /= *sw0.get_unchecked(g) };
    }
}

/// Run 2-FE acceleration loop (demean_acc_gnl with two_fe=true).
fn run_2fe_acceleration(
    fe_info: &FEInfo,
    in_out: &[f64],
    alpha: &mut [f64],      // Current coefficients, modified in place
    beta: &mut [f64],       // Temporary buffer
    config: &FixestConfig,
    max_iter: usize,
) -> (usize, bool) {
    let n0 = fe_info.n_groups[0];

    // Working buffers
    let mut gx = vec![0.0; n0];
    let mut ggx = vec![0.0; n0];
    let mut temp = vec![0.0; n0];
    let mut beta_tmp = vec![0.0; fe_info.n_groups[1]];

    // Grand acceleration buffers
    let mut y = vec![0.0; n0];
    let mut gy = vec![0.0; n0];
    let mut ggy = vec![0.0; n0];
    let mut grand_counter = 0usize;

    // First iteration: G(alpha)
    project_2fe(fe_info, in_out, alpha, &mut gx, beta);

    let mut keep_going = should_continue(alpha, &gx, config.tol);
    let mut iter = 0;

    while keep_going && iter < max_iter {
        iter += 1;

        // G(G(alpha))
        project_2fe(fe_info, in_out, &gx, &mut ggx, &mut beta_tmp);

        // Irons-Tuck
        if irons_tuck_update(alpha, &gx, &ggx) {
            break;
        }

        // Project after acceleration
        if iter >= config.iter_proj_after_acc {
            temp.copy_from_slice(alpha);
            project_2fe(fe_info, in_out, &temp, alpha, &mut beta_tmp);
        }

        // G(alpha)
        project_2fe(fe_info, in_out, alpha, &mut gx, beta);

        // Convergence check
        keep_going = should_continue(alpha, &gx, config.tol);

        // Grand acceleration
        if iter % config.iter_grand_acc == 0 {
            grand_counter += 1;
            match grand_counter {
                1 => y.copy_from_slice(&gx),
                2 => gy.copy_from_slice(&gx),
                _ => {
                    ggy.copy_from_slice(&gx);
                    if irons_tuck_update(&mut y, &gy, &ggy) {
                        break;
                    }
                    project_2fe(fe_info, in_out, &y, &mut gx, beta);
                    grand_counter = 0;
                }
            }
        }
    }

    (iter, !keep_going)
}

// =============================================================================
// General Q-FE Coefficient-Space Implementation (matching compute_fe_gnl)
// =============================================================================

/// Q-FE projection: Compute G(coef_in) -> coef_out.
/// Updates FEs in reverse order (Q-1 down to 0) matching fixest.
#[inline(always)]
fn project_qfe(
    fe_info: &FEInfo,
    in_out: &[f64],
    coef_in: &[f64],
    coef_out: &mut [f64],
    sum_other_means: &mut [f64], // N-length buffer
) {
    let n_fe = fe_info.n_fe;
    let n_obs = fe_info.n_obs;
    let weights = &fe_info.weights;

    // Process in reverse order
    for q in (0..n_fe).rev() {
        // Step 1: Compute sum of other FE contributions (NO weights here - this is just
        // expanding coefficients to observation space)
        sum_other_means.fill(0.0);

        // Add contributions from FEs with h < q (use coef_in)
        for h in 0..q {
            let start_h = fe_info.coef_start[h];
            let fe_h = &fe_info.fe_ids[h];
            // SAFETY: fe_h[i] < n_groups[h], start_h + fe_h[i] < coef_in.len()
            for i in 0..n_obs {
                unsafe {
                    let g = *fe_h.get_unchecked(i);
                    *sum_other_means.get_unchecked_mut(i) += *coef_in.get_unchecked(start_h + g);
                }
            }
        }

        // Add contributions from FEs with h > q (use coef_out, already computed)
        for h in (q + 1)..n_fe {
            let start_h = fe_info.coef_start[h];
            let fe_h = &fe_info.fe_ids[h];
            // SAFETY: fe_h[i] < n_groups[h], start_h + fe_h[i] < coef_out.len()
            for i in 0..n_obs {
                unsafe {
                    let g = *fe_h.get_unchecked(i);
                    *sum_other_means.get_unchecked_mut(i) += *coef_out.get_unchecked(start_h + g);
                }
            }
        }

        // Step 2: Compute new coefficients for FE q
        let start_q = fe_info.coef_start[q];
        let n_groups_q = fe_info.n_groups[q];
        let fe_q = &fe_info.fe_ids[q];
        let sw_q = &fe_info.sum_weights[q];

        // Initialize to in_out (pre-aggregated weighted (input-output))
        coef_out[start_q..start_q + n_groups_q]
            .copy_from_slice(&in_out[start_q..start_q + n_groups_q]);

        // Subtract weighted other FE contributions (weights applied when aggregating back)
        // SAFETY: fe_q[i] < n_groups_q, start_q + fe_q[i] < coef_out.len()
        if fe_info.is_unweighted {
            for i in 0..n_obs {
                unsafe {
                    let g = *fe_q.get_unchecked(i);
                    *coef_out.get_unchecked_mut(start_q + g) -= *sum_other_means.get_unchecked(i);
                }
            }
        } else {
            for i in 0..n_obs {
                unsafe {
                    let g = *fe_q.get_unchecked(i);
                    *coef_out.get_unchecked_mut(start_q + g) -=
                        *sum_other_means.get_unchecked(i) * *weights.get_unchecked(i);
                }
            }
        }

        // Divide by sum of weights
        for g in 0..n_groups_q {
            unsafe {
                *coef_out.get_unchecked_mut(start_q + g) /= *sw_q.get_unchecked(g);
            }
        }
    }
}

/// Run Q-FE acceleration loop (demean_acc_gnl).
#[allow(dead_code)]
fn run_qfe_acceleration(
    fe_info: &FEInfo,
    in_out: &[f64],
    coef: &mut [f64],       // Current coefficients, modified in place
    config: &FixestConfig,
    max_iter: usize,
    input: &[f64],          // Original input for SSR
) -> (usize, bool) {
    let n_coef = fe_info.n_coef_total;
    let n_obs = fe_info.n_obs;

    // nb_coef_no_Q: all except last FE (what fixest uses for acceleration)
    let nb_coef_no_q = n_coef - fe_info.n_groups[fe_info.n_fe - 1];

    // Working buffers
    let mut gx = vec![0.0; n_coef];
    let mut ggx = vec![0.0; n_coef];
    let mut temp = vec![0.0; n_coef];
    let mut sum_other_means = vec![0.0; n_obs];

    // Grand acceleration buffers (only nb_coef_no_q needed)
    let mut y = vec![0.0; n_coef];
    let mut gy = vec![0.0; n_coef];
    let mut ggy = vec![0.0; n_coef];
    let mut grand_counter = 0usize;

    // SSR buffer
    let mut output_buf = vec![0.0; n_obs];
    let mut ssr = 0.0;

    // First iteration: G(coef)
    project_qfe(fe_info, in_out, coef, &mut gx, &mut sum_other_means);

    let mut keep_going = should_continue(&coef[..nb_coef_no_q], &gx[..nb_coef_no_q], config.tol);
    let mut iter = 0;

    while keep_going && iter < max_iter {
        iter += 1;

        // G(G(coef))
        project_qfe(fe_info, in_out, &gx, &mut ggx, &mut sum_other_means);

        // Irons-Tuck on nb_coef_no_q
        if irons_tuck_update(&mut coef[..nb_coef_no_q], &gx[..nb_coef_no_q], &ggx[..nb_coef_no_q]) {
            break;
        }

        // Project after acceleration
        if iter >= config.iter_proj_after_acc {
            temp.copy_from_slice(coef);
            project_qfe(fe_info, in_out, &temp, coef, &mut sum_other_means);
        }

        // G(coef)
        project_qfe(fe_info, in_out, coef, &mut gx, &mut sum_other_means);

        // Convergence check on nb_coef_no_q
        keep_going = should_continue(&coef[..nb_coef_no_q], &gx[..nb_coef_no_q], config.tol);

        // Grand acceleration on nb_coef_no_q
        if iter % config.iter_grand_acc == 0 {
            grand_counter += 1;
            match grand_counter {
                1 => y[..nb_coef_no_q].copy_from_slice(&gx[..nb_coef_no_q]),
                2 => gy[..nb_coef_no_q].copy_from_slice(&gx[..nb_coef_no_q]),
                _ => {
                    ggy[..nb_coef_no_q].copy_from_slice(&gx[..nb_coef_no_q]);
                    if irons_tuck_update(&mut y[..nb_coef_no_q], &gy[..nb_coef_no_q], &ggy[..nb_coef_no_q]) {
                        break;
                    }
                    project_qfe(fe_info, in_out, &y, &mut gx, &mut sum_other_means);
                    grand_counter = 0;
                }
            }
        }

        // SSR stopping every 40 iterations
        if iter % 40 == 0 {
            let ssr_old = ssr;
            fe_info.compute_output(&gx, input, &mut output_buf);
            ssr = output_buf.iter().map(|&r| r * r).sum();

            if iter > 40 && stopping_crit(ssr_old, ssr, config.tol) {
                break;
            }
        }
    }

    // Copy final gx to coef
    coef.copy_from_slice(&gx);

    (iter, !keep_going)
}

// =============================================================================
// Public API: demean_single matching fixest's demean_single_gnl
// =============================================================================

/// Demean a single variable using coefficient-space iteration.
/// Matches fixest's demean_single_gnl exactly.
pub fn demean_single(
    fe_info: &FEInfo,
    input: &[f64],
    config: &FixestConfig,
) -> (Vec<f64>, usize, bool) {
    let n_obs = fe_info.n_obs;
    let n_fe = fe_info.n_fe;

    // Output initialized to 0
    let mut output = vec![0.0; n_obs];

    // Compute initial in_out
    let in_out = fe_info.compute_in_out(input, &output);

    if n_fe == 1 {
        // Single FE: closed-form solution
        let mut result = vec![0.0; n_obs];
        let fe0 = &fe_info.fe_ids[0];
        let sw0 = &fe_info.sum_weights[0];

        // coef[g] = in_out[g] / sw[g]
        let coef: Vec<f64> = in_out.iter().zip(sw0.iter()).map(|(&io, &sw)| io / sw).collect();

        // output[i] = input[i] - coef[fe0[i]]
        for i in 0..n_obs {
            result[i] = input[i] - coef[fe0[i]];
        }

        return (result, 0, true);
    }

    if n_fe == 2 {
        // 2-FE: Use specialized 2-FE algorithm
        let n0 = fe_info.n_groups[0];
        let n1 = fe_info.n_groups[1];

        let mut alpha = vec![0.0; n0];
        let mut beta = vec![0.0; n1];

        let (iter, converged) = run_2fe_acceleration(
            fe_info,
            &in_out,
            &mut alpha,
            &mut beta,
            config,
            config.maxiter,
        );

        // Compute output
        let mut result = vec![0.0; n_obs];
        let fe0 = &fe_info.fe_ids[0];
        let fe1 = &fe_info.fe_ids[1];

        for i in 0..n_obs {
            result[i] = input[i] - alpha[fe0[i]] - beta[fe1[i]];
        }

        return (result, iter, converged);
    }

    // 3+ FE: Use fixest's multi-phase strategy
    // Key insight: fixest's output stores SUM OF FE COEFFICIENTS, not residual.
    // in_out = agg(input - output) = agg(input - sum_of_coefs) = agg(residual)
    // We'll use mu to store sum of FE coefs, then convert to residual at the end.
    //
    // 1. Warmup iterations on all FEs
    // 2. 2-FE sub-convergence on first 2 FEs
    // 3. Re-acceleration on all FEs

    let n_coef = fe_info.n_coef_total;
    let n0 = fe_info.n_groups[0];
    let n1 = fe_info.n_groups[1];
    let mut total_iter = 0usize;

    // mu = sum of FE contributions per observation (fixest's "output")
    // Starts at 0, accumulates FE coefficients across phases
    let mut mu = vec![0.0; n_obs];

    // Helper to compute in_out = agg(input - mu) per FE group
    let compute_in_out_from_mu = |mu: &[f64]| -> Vec<f64> {
        let mut in_out = vec![0.0; fe_info.n_coef_total];
        for q in 0..fe_info.n_fe {
            let start = fe_info.coef_start[q];
            let fe_q = &fe_info.fe_ids[q];
            if fe_info.is_unweighted {
                for i in 0..n_obs {
                    in_out[start + fe_q[i]] += input[i] - mu[i];
                }
            } else {
                for i in 0..n_obs {
                    in_out[start + fe_q[i]] += (input[i] - mu[i]) * fe_info.weights[i];
                }
            }
        }
        in_out
    };

    // Helper to add coefficients to mu
    let add_coef_to_mu = |coef: &[f64], mu: &mut [f64]| {
        for q in 0..fe_info.n_fe {
            let start = fe_info.coef_start[q];
            let fe_q = &fe_info.fe_ids[q];
            for i in 0..n_obs {
                mu[i] += coef[start + fe_q[i]];
            }
        }
    };

    // Phase 1: Warmup with all FEs
    let mut coef = vec![0.0; n_coef];
    let in_out_phase1 = compute_in_out_from_mu(&mu);

    let (iter1, converged1) = run_qfe_acceleration(
        fe_info,
        &in_out_phase1,
        &mut coef,
        config,
        config.iter_warmup,
        input,
    );
    total_iter += iter1;

    // Add Phase 1 coefficients to mu
    add_coef_to_mu(&coef, &mut mu);

    if !converged1 {
        // Phase 2: 2-FE sub-convergence on first 2 FEs
        let in_out_phase2 = compute_in_out_from_mu(&mu);

        // Start with fresh alpha, beta
        let mut alpha = vec![0.0; n0];
        let mut beta = vec![0.0; n1];

        // Extract only the first 2 FE portions of in_out
        let in_out_2fe: Vec<f64> = in_out_phase2[..n0 + n1].to_vec();

        let iter_max_2fe = config.maxiter / 2;
        let (iter2, _) = run_2fe_acceleration(
            fe_info,
            &in_out_2fe,
            &mut alpha,
            &mut beta,
            config,
            iter_max_2fe,
        );
        total_iter += iter2;

        // Add Phase 2's alpha/beta to mu (only FE0 and FE1)
        let fe0 = &fe_info.fe_ids[0];
        let fe1 = &fe_info.fe_ids[1];
        for i in 0..n_obs {
            mu[i] += alpha[fe0[i]] + beta[fe1[i]];
        }

        // Phase 3: Re-acceleration on all FEs
        let remaining = config.maxiter.saturating_sub(total_iter);
        if remaining > 0 {
            let in_out_phase3 = compute_in_out_from_mu(&mu);

            // Start with fresh coefficients
            coef.fill(0.0);

            let (iter3, _) = run_qfe_acceleration(
                fe_info,
                &in_out_phase3,
                &mut coef,
                config,
                remaining,
                input,
            );
            total_iter += iter3;

            // Add Phase 3 coefficients to mu
            add_coef_to_mu(&coef, &mut mu);
        }
    }

    // Convert mu (sum of FE coefs) to output (residual = input - mu)
    for i in 0..n_obs {
        output[i] = input[i] - mu[i];
    }

    let converged = total_iter < config.maxiter;
    (output, total_iter, converged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2fe_convergence() {
        let n_obs = 100;
        let n_fe = 2;

        // Create simple FE structure
        let mut group_ids = Vec::with_capacity(n_obs * n_fe);
        for i in 0..n_obs {
            group_ids.push(i % 10);  // FE1: 10 groups
            group_ids.push(i % 5);   // FE2: 5 groups
        }

        let n_groups = vec![10, 5];
        let weights = vec![1.0; n_obs];

        let fe_info = FEInfo::new(n_obs, n_fe, &group_ids, &n_groups, &weights);

        // Random input
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let (result, iter, converged) = demean_single(&fe_info, &input, &config);

        assert!(converged, "Should converge");
        assert!(iter < 100, "Should converge quickly");
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_3fe_convergence() {
        let n_obs = 100;
        let n_fe = 3;

        let mut group_ids = Vec::with_capacity(n_obs * n_fe);
        for i in 0..n_obs {
            group_ids.push(i % 10);  // FE1
            group_ids.push(i % 5);   // FE2
            group_ids.push(i % 3);   // FE3
        }

        let n_groups = vec![10, 5, 3];
        let weights = vec![1.0; n_obs];

        let fe_info = FEInfo::new(n_obs, n_fe, &group_ids, &n_groups, &weights);
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let (result, _iter, converged) = demean_single(&fe_info, &input, &config);

        assert!(converged);
        assert!(result.iter().all(|&v| v.is_finite()));
    }
}
