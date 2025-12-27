//! Coefficient-space demeaning matching fixest's algorithm exactly.
//!
//! This is a direct port of fixest's demeaning.cpp, using coefficient-space
//! iteration rather than residual-space iteration.

/// Pre-computed FE information for coefficient-space iteration.
/// Uses flat memory layout for better cache performance.
pub struct FEInfo {
    pub n_obs: usize,
    pub n_fe: usize,
    /// Group IDs flattened: fe_ids[q * n_obs + i] = group ID for observation i in FE q
    /// This eliminates pointer indirection compared to Vec<Vec<usize>>
    pub fe_ids: Vec<usize>,
    /// Number of groups per FE
    pub n_groups: Vec<usize>,
    /// Starting index of each FE's coefficients in coef array
    pub coef_start: Vec<usize>,
    /// Total number of coefficients
    pub n_coef_total: usize,
    /// Sum of weights per group, flattened: access via coef_start[q] + g
    pub sum_weights: Vec<f64>,
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

        // Coefficient starting indices (computed first, used for sum_weights layout)
        let mut coef_start = vec![0usize; n_fe];
        for q in 1..n_fe {
            coef_start[q] = coef_start[q - 1] + n_groups[q - 1];
        }
        let n_coef_total: usize = n_groups.iter().sum();

        // Flatten fe_ids: fe_ids[q * n_obs + i] = group_ids[i * n_fe + q]
        // This converts from row-major input to column-major (per-FE) layout
        let mut fe_ids = vec![0usize; n_fe * n_obs];
        for i in 0..n_obs {
            for q in 0..n_fe {
                fe_ids[q * n_obs + i] = group_ids[i * n_fe + q];
            }
        }

        // Sum of weights per group, flattened with same layout as coef
        let mut sum_weights = vec![0.0; n_coef_total];
        for q in 0..n_fe {
            let start = coef_start[q];
            let fe_offset = q * n_obs;
            for i in 0..n_obs {
                let g = fe_ids[fe_offset + i];
                sum_weights[start + g] += weights[i];
            }
        }
        // Avoid division by zero
        for s in &mut sum_weights {
            if *s == 0.0 {
                *s = 1.0;
            }
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

    /// Get slice of FE group IDs for FE q: &[group_id for obs 0..n_obs]
    #[inline(always)]
    pub fn fe_ids_slice(&self, q: usize) -> &[usize] {
        let start = q * self.n_obs;
        &self.fe_ids[start..start + self.n_obs]
    }

    /// Get slice of sum_weights for FE q: &[sum_weight for group 0..n_groups[q]]
    #[inline(always)]
    pub fn sum_weights_slice(&self, q: usize) -> &[f64] {
        let start = self.coef_start[q];
        let end = if q + 1 < self.n_fe {
            self.coef_start[q + 1]
        } else {
            self.n_coef_total
        };
        &self.sum_weights[start..end]
    }

    /// Compute sum of weighted (input - output) for each coefficient.
    /// This is computed ONCE at the start and never changes.
    pub fn compute_in_out(&self, input: &[f64], output: &[f64]) -> Vec<f64> {
        let mut in_out = vec![0.0; self.n_coef_total];
        let n_obs = self.n_obs;

        if self.is_unweighted {
            for q in 0..self.n_fe {
                let start = self.coef_start[q];
                let fe_offset = q * n_obs;
                for i in 0..n_obs {
                    let g = self.fe_ids[fe_offset + i];
                    in_out[start + g] += input[i] - output[i];
                }
            }
        } else {
            for q in 0..self.n_fe {
                let start = self.coef_start[q];
                let fe_offset = q * n_obs;
                for i in 0..n_obs {
                    let g = self.fe_ids[fe_offset + i];
                    in_out[start + g] += (input[i] - output[i]) * self.weights[i];
                }
            }
        }

        in_out
    }

    /// Compute output from coefficients: output[i] = input[i] - sum_q(coef[fe_q[i]])
    pub fn compute_output(&self, coef: &[f64], input: &[f64], output: &mut [f64]) {
        output.copy_from_slice(input);
        let n_obs = self.n_obs;
        for q in 0..self.n_fe {
            let start = self.coef_start[q];
            let fe_offset = q * n_obs;
            for i in 0..n_obs {
                let g = self.fe_ids[fe_offset + i];
                output[i] -= coef[start + g];
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
            tol: 1e-6, // Match fixest's default
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
    let fe0 = fe_info.fe_ids_slice(0);
    let fe1 = fe_info.fe_ids_slice(1);
    let sw0 = fe_info.sum_weights_slice(0);
    let sw1 = fe_info.sum_weights_slice(1);
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

/// Compute beta from alpha (half of project_2fe, for SSR computation).
/// This matches fixest's compute_fe_coef_2_internal with step_2=false.
#[inline(always)]
fn compute_beta_from_alpha(
    fe_info: &FEInfo,
    in_out: &[f64],
    alpha: &[f64],
    beta: &mut [f64],
) {
    let n1 = fe_info.n_groups[1];
    let n_obs = fe_info.n_obs;
    let n0 = fe_info.n_groups[0];
    let fe0 = fe_info.fe_ids_slice(0);
    let fe1 = fe_info.fe_ids_slice(1);
    let sw1 = fe_info.sum_weights_slice(1);
    let weights = &fe_info.weights;

    // beta[g] = (in_out[n0+g] - sum_{i:fe1[i]=g} alpha[fe0[i]] * w[i]) / sw1[g]
    beta[..n1].copy_from_slice(&in_out[n0..n0 + n1]);

    if fe_info.is_unweighted {
        for i in 0..n_obs {
            unsafe {
                let g1 = *fe1.get_unchecked(i);
                let g0 = *fe0.get_unchecked(i);
                *beta.get_unchecked_mut(g1) -= *alpha.get_unchecked(g0);
            }
        }
    } else {
        for i in 0..n_obs {
            unsafe {
                let g1 = *fe1.get_unchecked(i);
                let g0 = *fe0.get_unchecked(i);
                *beta.get_unchecked_mut(g1) -= *alpha.get_unchecked(g0) * *weights.get_unchecked(i);
            }
        }
    }

    for g in 0..n1 {
        unsafe { *beta.get_unchecked_mut(g) /= *sw1.get_unchecked(g) };
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
    input: &[f64],          // Original input for SSR stopping criterion
) -> (usize, bool) {
    let n0 = fe_info.n_groups[0];
    let n1 = fe_info.n_groups[1];
    let n_obs = fe_info.n_obs;

    // Working buffers
    let mut gx = vec![0.0; n0];
    let mut ggx = vec![0.0; n0];
    let mut temp = vec![0.0; n0];
    let mut beta_tmp = vec![0.0; n1];

    // Grand acceleration buffers
    let mut y = vec![0.0; n0];
    let mut gy = vec![0.0; n0];
    let mut ggy = vec![0.0; n0];
    let mut grand_counter = 0usize;

    // SSR tracking
    let mut ssr = 0.0;
    let fe0 = fe_info.fe_ids_slice(0);
    let fe1 = fe_info.fe_ids_slice(1);

    // First iteration: G(alpha)
    project_2fe(fe_info, in_out, alpha, &mut gx, beta);

    let mut keep_going = should_continue(alpha, &gx, config.tol);
    let mut iter = 0;

    if std::env::var("PYFIXEST_DEBUG_ITER").is_ok() {
        let alpha_norm: f64 = alpha.iter().map(|x| x * x).sum();
        let gx_norm: f64 = gx.iter().map(|x| x * x).sum();
        let diff_norm: f64 = alpha.iter().zip(gx.iter()).map(|(a, g)| (a - g).powi(2)).sum();
        eprintln!("[run_2fe_acc] Initial: alpha_norm={:.6e}, gx_norm={:.6e}, diff_norm={:.6e}, keep_going={}",
                  alpha_norm, gx_norm, diff_norm, keep_going);
    }

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

        // SSR stopping criterion every 40 iterations (matching fixest)
        if iter % 40 == 0 {
            let ssr_old = ssr;

            // Compute beta from gx (current alpha) for SSR computation
            // Only need to compute beta, not full projection (matches fixest)
            compute_beta_from_alpha(fe_info, in_out, &gx, &mut beta_tmp);

            // Compute SSR = sum((input - alpha[fe0] - beta[fe1])^2)
            ssr = 0.0;
            for i in 0..n_obs {
                let resid = input[i] - gx[fe0[i]] - beta_tmp[fe1[i]];
                ssr += resid * resid;
            }

            if iter > 40 && stopping_crit(ssr_old, ssr, config.tol) {
                break;
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
/// Specialized for 3 FEs (most common case) with loop unrolling.
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

    // Pre-compute raw pointers for hot loops
    let fe_ids_ptr = fe_info.fe_ids.as_ptr();
    let coef_start = &fe_info.coef_start;
    let sum_other_ptr = sum_other_means.as_mut_ptr();
    let coef_in_ptr = coef_in.as_ptr();
    let coef_out_ptr = coef_out.as_mut_ptr();
    let weights_ptr = fe_info.weights.as_ptr();

    // Specialized fast path for 3 FEs (common case)
    if n_fe == 3 && fe_info.is_unweighted {
        project_qfe_3fe_unweighted(
            n_obs,
            fe_ids_ptr,
            coef_start,
            sum_other_ptr,
            coef_in_ptr,
            coef_out_ptr,
            in_out,
            &fe_info.n_groups,
            &fe_info.sum_weights,
        );
        return;
    }

    // General case for any number of FEs
    project_qfe_general(
        fe_info,
        in_out,
        coef_in,
        coef_out,
        sum_other_means,
        n_fe,
        n_obs,
        fe_ids_ptr,
        coef_start,
        sum_other_ptr,
        coef_in_ptr,
        coef_out_ptr,
        weights_ptr,
    );
}

/// Specialized 3-FE projection for unweighted case.
#[inline(always)]
fn project_qfe_3fe_unweighted(
    n_obs: usize,
    fe_ids_ptr: *const usize,
    coef_start: &[usize],
    sum_other_ptr: *mut f64,
    coef_in_ptr: *const f64,
    coef_out_ptr: *mut f64,
    in_out: &[f64],
    n_groups: &[usize],
    sum_weights: &[f64],
) {
    let (start_0, start_1, start_2) = (coef_start[0], coef_start[1], coef_start[2]);
    let fe_0_ptr = fe_ids_ptr;
    let fe_1_ptr = unsafe { fe_ids_ptr.add(n_obs) };
    let fe_2_ptr = unsafe { fe_ids_ptr.add(2 * n_obs) };
    let in_out_ptr = in_out.as_ptr();

    // === q=2: Process FE 2 (add from FE 0, 1 using coef_in) ===
    // No need to fill with zeros - we directly assign the sum of FE 0 and FE 1 contributions
    // Unrolled loop: process 4 observations at a time
    let n_chunks = n_obs / 4;
    let remainder = n_obs % 4;

    unsafe {
        for chunk in 0..n_chunks {
            let base = chunk * 4;
            let g0_0 = *fe_0_ptr.add(base);
            let g0_1 = *fe_0_ptr.add(base + 1);
            let g0_2 = *fe_0_ptr.add(base + 2);
            let g0_3 = *fe_0_ptr.add(base + 3);
            let g1_0 = *fe_1_ptr.add(base);
            let g1_1 = *fe_1_ptr.add(base + 1);
            let g1_2 = *fe_1_ptr.add(base + 2);
            let g1_3 = *fe_1_ptr.add(base + 3);

            *sum_other_ptr.add(base) =
                *coef_in_ptr.add(start_0 + g0_0) + *coef_in_ptr.add(start_1 + g1_0);
            *sum_other_ptr.add(base + 1) =
                *coef_in_ptr.add(start_0 + g0_1) + *coef_in_ptr.add(start_1 + g1_1);
            *sum_other_ptr.add(base + 2) =
                *coef_in_ptr.add(start_0 + g0_2) + *coef_in_ptr.add(start_1 + g1_2);
            *sum_other_ptr.add(base + 3) =
                *coef_in_ptr.add(start_0 + g0_3) + *coef_in_ptr.add(start_1 + g1_3);
        }

        for i in (n_chunks * 4)..(n_chunks * 4 + remainder) {
            let g0 = *fe_0_ptr.add(i);
            let g1 = *fe_1_ptr.add(i);
            *sum_other_ptr.add(i) = *coef_in_ptr.add(start_0 + g0) + *coef_in_ptr.add(start_1 + g1);
        }
    }

    // Compute coef_out for FE 2
    let n_groups_2 = n_groups[2];
    unsafe {
        std::ptr::copy_nonoverlapping(
            in_out_ptr.add(start_2),
            coef_out_ptr.add(start_2),
            n_groups_2,
        );
    }

    unsafe {
        for i in 0..n_obs {
            let g = *fe_2_ptr.add(i);
            *coef_out_ptr.add(start_2 + g) -= *sum_other_ptr.add(i);
        }
        for g in 0..n_groups_2 {
            *coef_out_ptr.add(start_2 + g) /= *sum_weights.get_unchecked(start_2 + g);
        }
    }

    // === q=1: Process FE 1 (add from FE 0 using coef_in, FE 2 using coef_out) ===
    unsafe {
        for chunk in 0..n_chunks {
            let base = chunk * 4;
            let g0_0 = *fe_0_ptr.add(base);
            let g0_1 = *fe_0_ptr.add(base + 1);
            let g0_2 = *fe_0_ptr.add(base + 2);
            let g0_3 = *fe_0_ptr.add(base + 3);
            let g2_0 = *fe_2_ptr.add(base);
            let g2_1 = *fe_2_ptr.add(base + 1);
            let g2_2 = *fe_2_ptr.add(base + 2);
            let g2_3 = *fe_2_ptr.add(base + 3);

            *sum_other_ptr.add(base) =
                *coef_in_ptr.add(start_0 + g0_0) + *coef_out_ptr.add(start_2 + g2_0);
            *sum_other_ptr.add(base + 1) =
                *coef_in_ptr.add(start_0 + g0_1) + *coef_out_ptr.add(start_2 + g2_1);
            *sum_other_ptr.add(base + 2) =
                *coef_in_ptr.add(start_0 + g0_2) + *coef_out_ptr.add(start_2 + g2_2);
            *sum_other_ptr.add(base + 3) =
                *coef_in_ptr.add(start_0 + g0_3) + *coef_out_ptr.add(start_2 + g2_3);
        }

        for i in (n_chunks * 4)..(n_chunks * 4 + remainder) {
            let g0 = *fe_0_ptr.add(i);
            let g2 = *fe_2_ptr.add(i);
            *sum_other_ptr.add(i) = *coef_in_ptr.add(start_0 + g0) + *coef_out_ptr.add(start_2 + g2);
        }
    }

    // Compute coef_out for FE 1
    let n_groups_1 = n_groups[1];
    unsafe {
        std::ptr::copy_nonoverlapping(
            in_out_ptr.add(start_1),
            coef_out_ptr.add(start_1),
            n_groups_1,
        );
    }

    unsafe {
        for i in 0..n_obs {
            let g = *fe_1_ptr.add(i);
            *coef_out_ptr.add(start_1 + g) -= *sum_other_ptr.add(i);
        }
        for g in 0..n_groups_1 {
            *coef_out_ptr.add(start_1 + g) /= *sum_weights.get_unchecked(start_1 + g);
        }
    }

    // === q=0: Process FE 0 (add from FE 1, 2 using coef_out) ===
    unsafe {
        for chunk in 0..n_chunks {
            let base = chunk * 4;
            let g1_0 = *fe_1_ptr.add(base);
            let g1_1 = *fe_1_ptr.add(base + 1);
            let g1_2 = *fe_1_ptr.add(base + 2);
            let g1_3 = *fe_1_ptr.add(base + 3);
            let g2_0 = *fe_2_ptr.add(base);
            let g2_1 = *fe_2_ptr.add(base + 1);
            let g2_2 = *fe_2_ptr.add(base + 2);
            let g2_3 = *fe_2_ptr.add(base + 3);

            *sum_other_ptr.add(base) =
                *coef_out_ptr.add(start_1 + g1_0) + *coef_out_ptr.add(start_2 + g2_0);
            *sum_other_ptr.add(base + 1) =
                *coef_out_ptr.add(start_1 + g1_1) + *coef_out_ptr.add(start_2 + g2_1);
            *sum_other_ptr.add(base + 2) =
                *coef_out_ptr.add(start_1 + g1_2) + *coef_out_ptr.add(start_2 + g2_2);
            *sum_other_ptr.add(base + 3) =
                *coef_out_ptr.add(start_1 + g1_3) + *coef_out_ptr.add(start_2 + g2_3);
        }

        for i in (n_chunks * 4)..(n_chunks * 4 + remainder) {
            let g1 = *fe_1_ptr.add(i);
            let g2 = *fe_2_ptr.add(i);
            *sum_other_ptr.add(i) =
                *coef_out_ptr.add(start_1 + g1) + *coef_out_ptr.add(start_2 + g2);
        }
    }

    // Compute coef_out for FE 0
    let n_groups_0 = n_groups[0];
    unsafe {
        std::ptr::copy_nonoverlapping(in_out_ptr.add(start_0), coef_out_ptr.add(start_0), n_groups_0);
    }

    unsafe {
        for i in 0..n_obs {
            let g = *fe_0_ptr.add(i);
            *coef_out_ptr.add(start_0 + g) -= *sum_other_ptr.add(i);
        }
        for g in 0..n_groups_0 {
            *coef_out_ptr.add(start_0 + g) /= *sum_weights.get_unchecked(start_0 + g);
        }
    }
}

/// General Q-FE projection (any number of FEs, weighted or unweighted).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn project_qfe_general(
    fe_info: &FEInfo,
    in_out: &[f64],
    _coef_in: &[f64],  // Used via coef_in_ptr
    _coef_out: &mut [f64],  // Used via coef_out_ptr
    _sum_other_means: &mut [f64],  // Used via sum_other_ptr
    n_fe: usize,
    n_obs: usize,
    fe_ids_ptr: *const usize,
    coef_start: &[usize],
    sum_other_ptr: *mut f64,
    coef_in_ptr: *const f64,
    coef_out_ptr: *mut f64,
    weights_ptr: *const f64,
) {
    let in_out_ptr = in_out.as_ptr();

    // Process in reverse order (Q-1 down to 0, matching fixest)
    for q in (0..n_fe).rev() {
        // Step 1: Fill sum_other_means with zeros
        unsafe {
            std::ptr::write_bytes(sum_other_ptr, 0, n_obs);
        }

        // Add contributions from FEs with h < q (use coef_in)
        for h in 0..q {
            let start_h = coef_start[h];
            let fe_h_ptr = unsafe { fe_ids_ptr.add(h * n_obs) };
            for i in 0..n_obs {
                unsafe {
                    let g = *fe_h_ptr.add(i);
                    *sum_other_ptr.add(i) += *coef_in_ptr.add(start_h + g);
                }
            }
        }

        // Add contributions from FEs with h > q (use coef_out)
        for h in (q + 1)..n_fe {
            let start_h = coef_start[h];
            let fe_h_ptr = unsafe { fe_ids_ptr.add(h * n_obs) };
            for i in 0..n_obs {
                unsafe {
                    let g = *fe_h_ptr.add(i);
                    *sum_other_ptr.add(i) += *coef_out_ptr.add(start_h + g);
                }
            }
        }

        // Step 2: Compute new coefficients for FE q
        let start_q = coef_start[q];
        let n_groups_q = fe_info.n_groups[q];
        let fe_q_ptr = unsafe { fe_ids_ptr.add(q * n_obs) };
        let sw_q = fe_info.sum_weights_slice(q);

        // Initialize to in_out
        unsafe {
            std::ptr::copy_nonoverlapping(
                in_out_ptr.add(start_q),
                coef_out_ptr.add(start_q),
                n_groups_q,
            );
        }

        // Subtract weighted other FE contributions
        if fe_info.is_unweighted {
            for i in 0..n_obs {
                unsafe {
                    let g = *fe_q_ptr.add(i);
                    *coef_out_ptr.add(start_q + g) -= *sum_other_ptr.add(i);
                }
            }
        } else {
            for i in 0..n_obs {
                unsafe {
                    let g = *fe_q_ptr.add(i);
                    *coef_out_ptr.add(start_q + g) -=
                        *sum_other_ptr.add(i) * *weights_ptr.add(i);
                }
            }
        }

        // Divide by sum of weights
        for g in 0..n_groups_q {
            unsafe {
                *coef_out_ptr.add(start_q + g) /= *sw_q.get_unchecked(g);
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
        let prev_keep_going = keep_going;
        keep_going = should_continue(&coef[..nb_coef_no_q], &gx[..nb_coef_no_q], config.tol);
        if std::env::var("PYFIXEST_DEBUG_ITER").is_ok() && prev_keep_going && !keep_going {
            eprintln!("[run_qfe_acc] Coefficient converged at iter {}", iter);
        }

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
                if std::env::var("PYFIXEST_DEBUG_ITER").is_ok() {
                    eprintln!("[run_qfe_acc] SSR converged at iter {}: ssr_old={:.6e}, ssr={:.6e}",
                              iter, ssr_old, ssr);
                }
                keep_going = false;  // Mark as converged
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
        let fe0 = fe_info.fe_ids_slice(0);
        let sw0 = fe_info.sum_weights_slice(0);

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
            input,
        );

        // Compute output
        let mut result = vec![0.0; n_obs];
        let fe0 = fe_info.fe_ids_slice(0);
        let fe1 = fe_info.fe_ids_slice(1);

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
            let fe_offset = q * n_obs;
            if fe_info.is_unweighted {
                for i in 0..n_obs {
                    let g = fe_info.fe_ids[fe_offset + i];
                    in_out[start + g] += input[i] - mu[i];
                }
            } else {
                for i in 0..n_obs {
                    let g = fe_info.fe_ids[fe_offset + i];
                    in_out[start + g] += (input[i] - mu[i]) * fe_info.weights[i];
                }
            }
        }
        in_out
    };

    // Helper to add coefficients to mu
    let add_coef_to_mu = |coef: &[f64], mu: &mut [f64]| {
        for q in 0..fe_info.n_fe {
            let start = fe_info.coef_start[q];
            let fe_offset = q * n_obs;
            for i in 0..n_obs {
                let g = fe_info.fe_ids[fe_offset + i];
                mu[i] += coef[start + g];
            }
        }
    };

    // Phase 1: Warmup with all FEs
    let mut coef = vec![0.0; n_coef];
    let in_out_phase1 = compute_in_out_from_mu(&mu);

    let t1 = std::time::Instant::now();
    let (iter1, converged1) = run_qfe_acceleration(
        fe_info,
        &in_out_phase1,
        &mut coef,
        config,
        config.iter_warmup,
        input,
    );
    let phase1_time = t1.elapsed();
    total_iter += iter1;

    // Debug: print iteration counts for 3+ FE case
    if std::env::var("PYFIXEST_DEBUG_ITER").is_ok() {
        eprintln!("[demean_single] Phase 1 (warmup): {} iters, converged={}, time={:.2}ms",
                  iter1, converged1, phase1_time.as_secs_f64() * 1000.0);
    }

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

        if std::env::var("PYFIXEST_DEBUG_ITER").is_ok() {
            let in_out_norm: f64 = in_out_2fe.iter().map(|x| x * x).sum();
            eprintln!("[demean_single] Phase 2: in_out_2fe norm^2={:.6e}, n0={}, n1={}",
                      in_out_norm, n0, n1);
        }

        // Compute effective input for SSR: input - mu (accounts for Phase 1)
        let effective_input: Vec<f64> = (0..n_obs).map(|i| input[i] - mu[i]).collect();

        let iter_max_2fe = config.maxiter / 2;
        let t2 = std::time::Instant::now();
        let (iter2, conv2) = run_2fe_acceleration(
            fe_info,
            &in_out_2fe,
            &mut alpha,
            &mut beta,
            config,
            iter_max_2fe,
            &effective_input,
        );
        let phase2_time = t2.elapsed();
        total_iter += iter2;

        if std::env::var("PYFIXEST_DEBUG_ITER").is_ok() {
            eprintln!("[demean_single] Phase 2 (2-FE): {} iters, converged={}, time={:.2}ms",
                      iter2, conv2, phase2_time.as_secs_f64() * 1000.0);
        }

        // Add Phase 2's alpha/beta to mu (only FE0 and FE1)
        let fe0 = fe_info.fe_ids_slice(0);
        let fe1 = fe_info.fe_ids_slice(1);
        for i in 0..n_obs {
            mu[i] += alpha[fe0[i]] + beta[fe1[i]];
        }

        // Phase 3: Re-acceleration on all FEs
        let remaining = config.maxiter.saturating_sub(total_iter);
        if remaining > 0 {
            let in_out_phase3 = compute_in_out_from_mu(&mu);

            // Start with fresh coefficients
            coef.fill(0.0);

            let t3 = std::time::Instant::now();
            let (iter3, conv3) = run_qfe_acceleration(
                fe_info,
                &in_out_phase3,
                &mut coef,
                config,
                remaining,
                input,
            );
            let phase3_time = t3.elapsed();
            total_iter += iter3;

            if std::env::var("PYFIXEST_DEBUG_ITER").is_ok() {
                eprintln!("[demean_single] Phase 3 (re-acc): {} iters, converged={}, time={:.2}ms",
                          iter3, conv3, phase3_time.as_secs_f64() * 1000.0);
            }

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
