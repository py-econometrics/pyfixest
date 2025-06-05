// build with: cargo flamegraph --bin profile --root
use rand::{Rng, SeedableRng};

use ndarray::{Array2, ArrayView1, ArrayView2, Zip};
//use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

mod internal {
    pub(super) fn sad_converged(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol)
    }

    pub(super) fn subtract_weighted_group_mean(
        x: &mut [f64],
        sample_weights: &[f64],
        group_ids: &[usize],
        group_weights: &[f64],
        group_weighted_sums: &mut [f64],
    ) {
        group_weighted_sums.fill(0.0);

        // Accumulate weighted sums per group
        x.iter()
            .zip(sample_weights)
            .zip(group_ids)
            .for_each(|((&xi, &wi), &gid)| {
                group_weighted_sums[gid] += wi * xi;
            });

        // Compute group means
        let group_means: Vec<f64> = group_weighted_sums
            .iter()
            .zip(group_weights)
            .map(|(&sum, &weight)| sum / weight)
            .collect();

        // Subtract means from each sample
        x.iter_mut().zip(group_ids).for_each(|(xi, &gid)| {
            *xi -= group_means[gid];
        });
    }

    pub(super) fn calc_group_weights(
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
}

pub fn demean_impl(
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
    let group_weights =
        internal::calc_group_weights(&sample_weights, &group_ids, n_samples, n_factors, n_groups);

    let not_converged = Arc::new(AtomicUsize::new(0));

    // Precompute slices of group_ids for each factor
    let group_ids_by_factor: Vec<Vec<usize>> = (0..n_factors)
        .map(|j| {
            (0..n_samples)
                .map(|i| group_ids[i * n_factors + j])
                .collect()
        })
        .collect();

    // Precompute group weight slices
    let group_weight_slices: Vec<&[f64]> = (0..n_factors)
        .map(|j| &group_weights[j * n_groups..(j + 1) * n_groups])
        .collect();

    let process_column = |(k, mut col): (usize, ndarray::ArrayViewMut1<f64>)| {
        let mut xk_curr: Vec<f64> = (0..n_samples).map(|i| x[[i, k]]).collect();
        let mut xk_prev: Vec<f64> = xk_curr.iter().map(|&v| v - 1.0).collect();
        let mut gw_sums = vec![0.0; n_groups];

        let mut converged = false;
        for _ in 0..maxiter {
            for j in 0..n_factors {
                internal::subtract_weighted_group_mean(
                    &mut xk_curr,
                    &sample_weights,
                    &group_ids_by_factor[j],
                    group_weight_slices[j],
                    &mut gw_sums,
                );
            }

            if internal::sad_converged(&xk_curr, &xk_prev, tol) {
                converged = true;
                break;
            }
            xk_prev.copy_from_slice(&xk_curr);
        }

        if !converged {
            not_converged.fetch_add(1, Ordering::SeqCst);
        }
        Zip::from(&mut col).and(&xk_curr).for_each(|col_elm, &val| {
            *col_elm = val;
        });
    };

    let mut res = Array2::<f64>::zeros((n_samples, n_features));

    res.axis_iter_mut(ndarray::Axis(1))
        .into_iter()
        .enumerate()
        .for_each(process_column);

    let success = not_converged.load(Ordering::SeqCst) == 0;
    (res, success)
}


fn main() {

    // --- your setup and benchmarking code ---
    const N: usize = 1_00_000;
    const SEED: u64 = 0x5eed;

    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);

    let x: ndarray::Array2<f64> = ndarray::Array2::from_shape_fn((N, 10), |_| rng.random::<f64>());
    let flist: ndarray::Array2<usize> = ndarray::Array2::from_shape_fn((N, 3), |(_, j)| match j {
        0 => rng.random_range(0..10_000),
        1 => rng.random_range(0..3_000),
        _ => rng.random_range(0..100),
    });
    let weights = ndarray::Array1::ones(N);

    let guard = pprof::ProfilerGuard::new(100).unwrap();
    let (out, success) = demean_impl(&x.view(), &flist.view(), &weights.view(), 1e-8, 100_000);

    eprintln!("Converged? {success}");
    std::hint::black_box(out);

    // --- save the flamegraph ---
    if let Ok(report) = guard.report().build() {
        use std::fs::File;
        let file = File::create("flamegraph.svg").unwrap();
        report.flamegraph(file).unwrap();
    } else {
        eprintln!("pprof: no report generated (maybe your code ran too fast?)");
    }
}
