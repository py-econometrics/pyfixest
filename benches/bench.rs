//! benches/bench.rs
//! • Run with:  `cargo bench`
//! • Profile with:  `cargo flamegraph --bin profile` (if you have that binary)

use criterion::{criterion_group, criterion_main, Criterion};
use rand::{Rng, SeedableRng};
use ndarray::{Array2, ArrayView1, ArrayView2, Zip};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// -------------------------------------------------------------
// Helper module – same maths, but row-parallel where needed
// -------------------------------------------------------------
mod internal {
    use rayon::prelude::*;

    pub(super) fn sad_converged(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol)
    }

    pub(super) fn calc_group_weights(
        sample_weights: &[f64],
        group_ids_flat: &[usize],
        n_samples: usize,
        n_factors: usize,
        n_groups: usize,
    ) -> Vec<f64> {
        let mut out = vec![0.0; n_factors * n_groups];
        for i in 0..n_samples {
            let w = sample_weights[i];
            for j in 0..n_factors {
                let gid = group_ids_flat[i * n_factors + j];
                out[j * n_groups + gid] += w;
            }
        }
        out
    }

    /// Subtract weighted group means – parallel over rows (N)
    /// Each thread accumulates into its own Vec, then we reduce.
    pub(super) fn subtract_weighted_group_mean_parallel(
        x: &mut [f64],
        sample_weights: &[f64],
        group_ids: &[usize],
        group_weights: &[f64],   // len == n_groups
        scratch_means: &mut [f64], // len == n_groups
    ) {
        let n_groups = scratch_means.len();

        // 1) Each thread produces a local Vec<f64> of length n_groups
        let sums = x
            .par_iter()
            .zip(sample_weights)
            .zip(group_ids)
            .fold(
                || vec![0.0f64; n_groups],
                |mut acc, ((&xi, &wi), &gid)| {
                    acc[gid] += wi * xi;
                    acc
                },
            )
            .reduce(
                || vec![0.0f64; n_groups],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b) {
                        *ai += bi;
                    }
                    a
                },
            );

        // 2) Convert sums → means
        for gid in 0..n_groups {
            scratch_means[gid] = sums[gid] / group_weights[gid];
        }

        // 3) Subtract means from each sample (also parallel)
        x.par_iter_mut()
            .zip(group_ids)
            .for_each(|(xi, &gid)| {
                *xi -= scratch_means[gid];
            });
    }
}

// -------------------------------------------------------------
// Core algorithm – serial over columns, row-parallel inside
// -------------------------------------------------------------
pub fn demean_impl(
    x: &ArrayView2<f64>,
    flist: &ArrayView2<usize>,
    weights: &ArrayView1<f64>,
    tol: f64,
    maxiter: usize,
) -> (Array2<f64>, bool) {
    let (n_samples, n_features) = x.dim();
    let n_factors = flist.ncols();
    let n_groups = flist.iter().copied().max().unwrap() + 1;

    let sample_weights: Vec<f64> = weights.iter().copied().collect();
    let group_ids_flat: Vec<usize> = flist.iter().copied().collect();
    let group_weights = internal::calc_group_weights(
        &sample_weights,
        &group_ids_flat,
        n_samples,
        n_factors,
        n_groups,
    );

    // Pre-slice group IDs for each factor (Vec<Vec<usize>> to simplify lifetimes)
    let group_ids_by_factor: Vec<Vec<usize>> = (0..n_factors)
        .map(|j| {
            (0..n_samples)
                .map(|i| group_ids_flat[i * n_factors + j])
                .collect()
        })
        .collect();

    let group_weight_slices: Vec<&[f64]> = (0..n_factors)
        .map(|j| &group_weights[j * n_groups..(j + 1) * n_groups])
        .collect();

    let not_converged = Arc::new(AtomicUsize::new(0));
    let mut res = Array2::<f64>::zeros((n_samples, n_features));

    // ----------------------------------------------------------------
    // Allocate scratch buffers ONCE, then reuse them for each column
    let mut scratch_curr = vec![0.0_f64; n_samples];
    let mut scratch_prev = vec![0.0_f64; n_samples];
    let mut group_means = vec![0.0_f64; n_groups];
    // ----------------------------------------------------------------

    for (k, mut col) in res.axis_iter_mut(ndarray::Axis(1)).enumerate() {
        // Copy column k into scratch_curr
        for (i, slot) in scratch_curr.iter_mut().enumerate() {
            *slot = x[[i, k]];
        }

        // Initialize scratch_prev to (curr - 1.0)
        scratch_prev.copy_from_slice(&scratch_curr);
        for v in &mut scratch_prev {
            *v -= 1.0;
        }

        let mut converged = false;
        for _ in 0..maxiter {
            for j in 0..n_factors {
                internal::subtract_weighted_group_mean_parallel(
                    &mut scratch_curr,
                    &sample_weights,
                    &group_ids_by_factor[j],
                    group_weight_slices[j],
                    &mut group_means,
                );
            }
            if internal::sad_converged(&scratch_curr, &scratch_prev, tol) {
                converged = true;
                break;
            }
            scratch_prev.copy_from_slice(&scratch_curr);
        }

        if !converged {
            not_converged.fetch_add(1, Ordering::SeqCst);
        }

        // Write results back to the k-th column of res
        Zip::from(&mut col).and(&scratch_curr).for_each(|c_elem, &v| {
            *c_elem = v;
        });
    }

    let mut res = Array2::<f64>::zeros((n_samples, n_features));

    res.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(process_column);

    let success = not_converged.load(Ordering::SeqCst) == 0;
    (res, success)
}

// -------------------------------------------------------------
// Criterion benchmark – sample_size(10) for quick runs
// -------------------------------------------------------------
fn bench_demean_impl(c: &mut Criterion) {
    const N: usize = 1_000_000;
    const SEED: u64 = 0x5eed;

    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let x: Array2<f64> = Array2::from_shape_fn((N, 10), |_| rng.random::<f64>());
    let flist: Array2<usize> = Array2::from_shape_fn((N, 3), |(_, j)| match j {
        0 => rng.random_range(0..10_000),
        1 => rng.random_range(0..3_000),
        _ => rng.random_range(0..100),
    });
    let weights = ndarray::Array1::ones(N);

    let mut group = c.benchmark_group("demean");
    group.sample_size(10);
    group.bench_function("demean_impl", |b| {
        b.iter(|| {
            let (out, success) =
                demean_impl(&x.view(), &flist.view(), &weights.view(), 1e-8, 100_000);
            std::hint::black_box(out);
            std::hint::black_box(success);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_demean_impl);
criterion_main!(benches);
