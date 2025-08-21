use ndarray::{Array2, ArrayView1, ArrayView2, Zip};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

mod internal {
    pub(super) fn sad_converged(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol)
    }
}

struct FactorDemeaner {
    sample_weights: Vec<f64>,
    group_ids: Vec<usize>,
    group_weights: Vec<f64>,
    group_weighted_sums: Vec<f64>,
}

impl FactorDemeaner {
    fn new(
        sample_weights: Vec<f64>,
        group_ids: Vec<usize>,
        group_weights: Vec<f64>,
        n_groups: usize,
    ) -> Self {
        Self {
            sample_weights,
            group_ids,
            group_weights,
            group_weighted_sums: vec![0.0; n_groups],
        }
    }

    fn project(&mut self, input: &[f64], output: &mut [f64]) {
        self.group_weighted_sums.fill(0.0);

        // Accumulate weighted sums per group
        input.iter()
            .zip(&self.sample_weights)
            .zip(&self.group_ids)
            .for_each(|((&xi, &wi), &gid)| {
                self.group_weighted_sums[gid] += wi * xi;
            });

        // Compute group means and write demeaned values to output
        output.iter_mut()
            .zip(&self.group_ids)
            .zip(input)
            .for_each(|((out, &gid), &inp)| {
                let group_mean = self.group_weighted_sums[gid] / self.group_weights[gid];
                *out = inp - group_mean;
            });
    }
}

struct MultiFactorDemeaner {
    factors: Vec<FactorDemeaner>,
    temp_buffer: Vec<f64>,
}

impl MultiFactorDemeaner {
    fn new(
        sample_weights: &[f64],
        group_ids: &[usize],
        n_samples: usize,
        n_factors: usize,
        n_groups: usize,
    ) -> Self {
        let group_weights = Self::calc_group_weights(sample_weights, group_ids, n_samples, n_factors, n_groups);
        let mut factors = Vec::new();

        for j in 0..n_factors {
            // Extract group IDs for this factor
            let factor_group_ids: Vec<usize> = (0..n_samples)
                .map(|i| group_ids[i * n_factors + j])
                .collect();

            // Extract group weights for this factor
            let factor_group_weights = group_weights[j * n_groups..(j + 1) * n_groups].to_vec();

            factors.push(FactorDemeaner::new(
                sample_weights.to_vec(),
                factor_group_ids,
                factor_group_weights,
                n_groups,
            ));
        }

        Self {
            factors,
            temp_buffer: vec![0.0; n_samples],
        }
    }

    fn calc_group_weights(
        sample_weights: &[f64],
        group_ids: &[usize],
        n_samples: usize,
        n_factors: usize,
        n_groups: usize,
    ) -> Vec<f64> {
        let mut group_weights = vec![0.0; n_factors * n_groups];
        for i in 0..n_samples {
            let weight = sample_weights[i];
            for j in 0..n_factors {
                let id = group_ids[i * n_factors + j];
                group_weights[j * n_groups + id] += weight;
            }
        }
        group_weights
    }

    fn project(&mut self, input: &[f64], output: &mut [f64]) {
        if self.factors.is_empty() {
            output.copy_from_slice(input);
            return;
        }

        // First projection: input -> output
        self.factors[0].project(input, output);

        // Sequential projections: alternate between output and temp_buffer
        let mut use_temp_as_input = false;
        for factor in self.factors.iter_mut().skip(1) {
            if use_temp_as_input {
                // temp_buffer -> output
                factor.project(&self.temp_buffer, output);
            } else {
                // output -> temp_buffer
                factor.project(output, &mut self.temp_buffer);
            }
            use_temp_as_input = !use_temp_as_input;
        }

        // If final result is in temp_buffer, copy to output
        if use_temp_as_input {
            output.copy_from_slice(&self.temp_buffer);
        }
    }
}

struct AccelerationBuffers {
    x_curr: Vec<f64>,       // Current iterate X
    gx_curr: Vec<f64>,      // G(X) - first projection
    ggx_curr: Vec<f64>,     // G(G(X)) - second projection
    delta_gx: Vec<f64>,     // GGX - GX (working buffer)
    delta2_x: Vec<f64>,     // acceleration calculation buffer
    x_prev: Vec<f64>,       // For convergence checking
}

impl AccelerationBuffers {
    fn new(n_samples: usize) -> Self {
        Self {
            x_curr: vec![0.0; n_samples],
            gx_curr: vec![0.0; n_samples],
            ggx_curr: vec![0.0; n_samples],
            delta_gx: vec![0.0; n_samples],
            delta2_x: vec![0.0; n_samples],
            x_prev: vec![0.0; n_samples],
        }
    }
}

struct IronTucksAcceleration {
    projection: MultiFactorDemeaner,
    buffers: AccelerationBuffers,
}

impl IronTucksAcceleration {
    fn new(projection: MultiFactorDemeaner, n_samples: usize) -> Self {
        Self {
            projection,
            buffers: AccelerationBuffers::new(n_samples),
        }
    }

    /// Single step of the fixed-point iteration with Irons-Tuck acceleration
    fn step(&mut self, should_accelerate: bool) -> bool {
        // Store previous for convergence check
        self.buffers.x_prev.copy_from_slice(&self.buffers.x_curr);

        if should_accelerate {
            // Apply acceleration every 3rd iteration (like in C++ code)
            self.irons_tuck_step()
        } else {
            // Regular projection step
            self.regular_step()
        }
    }

    fn regular_step(&mut self) -> bool {
        // G(X_curr) -> gx_curr, then gx_curr -> x_curr
        self.projection.project(&self.buffers.x_curr, &mut self.buffers.gx_curr);
        self.buffers.x_curr.copy_from_slice(&self.buffers.gx_curr);
        false // not converged via acceleration
    }

    fn irons_tuck_step(&mut self) -> bool {
        // Compute G(X) -> gx_curr
        self.projection.project(&self.buffers.x_curr, &mut self.buffers.gx_curr);

        // Compute G(G(X)) -> ggx_curr
        self.projection.project(&self.buffers.gx_curr, &mut self.buffers.ggx_curr);

        // Apply Irons-Tuck acceleration formula
        self.apply_acceleration()
    }

    fn apply_acceleration(&mut self) -> bool {
        let n = self.buffers.x_curr.len();

        // Compute delta_GX = GGX - GX and delta2_X = delta_GX - GX + X
        let mut vprod = 0.0;
        let mut ssq = 0.0;

        for i in 0..n {
            let gx_tmp = self.buffers.gx_curr[i];
            self.buffers.delta_gx[i] = self.buffers.ggx_curr[i] - gx_tmp;
            self.buffers.delta2_x[i] = self.buffers.delta_gx[i] - gx_tmp + self.buffers.x_curr[i];

            let delta2_x_tmp = self.buffers.delta2_x[i];
            vprod += self.buffers.delta_gx[i] * delta2_x_tmp;
            ssq += delta2_x_tmp * delta2_x_tmp;
        }

        if ssq == 0.0 {
            return true; // Numerically converged
        }

        let coef = vprod / ssq;

        // Update X: X = GGX - coef * delta_GX
        for i in 0..n {
            self.buffers.x_curr[i] = self.buffers.ggx_curr[i] - coef * self.buffers.delta_gx[i];
        }

        false
    }

    fn is_converged(&self, tol: f64) -> bool {
        internal::sad_converged(&self.buffers.x_curr, &self.buffers.x_prev, tol)
    }

    fn set_initial(&mut self, x: &[f64]) {
        self.buffers.x_curr.copy_from_slice(x);
    }

    fn get_result(&self) -> &[f64] {
        &self.buffers.x_curr
    }
}

fn demean_impl(
    x: &ArrayView2<f64>,
    flist: &ArrayView2<usize>,
    weights: &ArrayView1<f64>,
    tol: f64,
    maxiter: usize,
) -> (Array2<f64>, bool) {
    let (n_samples, n_features) = x.dim();
    let n_factors = flist.ncols();
    let n_groups = flist.iter().cloned().max().unwrap() + 1;

    let sample_weights: Vec<f64> = weights.iter().cloned().collect();
    let group_ids: Vec<usize> = flist.iter().cloned().collect();

    let not_converged = Arc::new(AtomicUsize::new(0));

    let process_column = |(k, mut col): (usize, ndarray::ArrayViewMut1<f64>)| {
        let xk_curr: Vec<f64> = (0..n_samples).map(|i| x[[i, k]]).collect();

        let demeaner = MultiFactorDemeaner::new(
            &sample_weights,
            &group_ids,
            n_samples,
            n_factors,
            n_groups,
        );

        let mut acceleration = IronTucksAcceleration::new(demeaner, n_samples);
        acceleration.set_initial(&xk_curr);

        let mut converged = false;
        for i in 0..maxiter {
            // Apply Irons-Tuck acceleration every 3rd iteration, regular projection otherwise
            let should_accelerate = i % 3 == 0 && i > 0;
            let num_converged = acceleration.step(should_accelerate);

            if num_converged || acceleration.is_converged(tol) {
                converged = true;
                break;
            }
        }

        if !converged {
            not_converged.fetch_add(1, Ordering::SeqCst);
        }

        Zip::from(&mut col).and(acceleration.get_result()).for_each(|col_elm, &val| {
            *col_elm = val;
        });
    };

    let mut res = Array2::<f64>::zeros((n_samples, n_features));

    res.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(process_column);

    let success = not_converged.load(Ordering::SeqCst) == 0;
    (res, success)
}

#[pyfunction]
#[pyo3(signature = (x, flist, weights, tol=1e-8, maxiter=100_000))]
pub fn _demean_accelerated_rs(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    flist: PyReadonlyArray2<usize>,
    weights: PyReadonlyArray1<f64>,
    tol: f64,
    maxiter: usize,
) -> PyResult<(Py<PyArray2<f64>>, bool)> {
    let x_arr = x.as_array();
    let flist_arr = flist.as_array();
    let weights_arr = weights.as_array();

    let (out, success) =
        py.allow_threads(|| demean_impl(&x_arr, &flist_arr, &weights_arr, tol, maxiter));

    let pyarray = PyArray2::from_owned_array(py, out);
    Ok((pyarray.into(), success))
}
