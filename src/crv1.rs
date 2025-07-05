use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

fn bucket_argsort_rs(arr: &ArrayView1<usize>) -> (Vec<usize>, Vec<usize>) {
    // 1. Count frequencies
    let maxv = *arr.iter().max().unwrap_or(&0);
    let counts = {
        let mut counts = vec![0usize; maxv + 1];
        arr.iter().for_each(|&v| counts[v] += 1);
        counts
    };

    // Compute the prefix sums of the counts vector
    let prefix_sum_iterator = counts.iter().scan(0, |acc, &count| {
        *acc += count;
        Some(*acc)
    });

    // Prepend the prefix sums with 0 and collect
    let locs: Vec<usize> = std::iter::once(0).chain(prefix_sum_iterator).collect();

    // 3. Copy locs to track insertion positions
    let mut pos = locs[..counts.len()].to_vec();

    // 4. Build argsort result
    let mut args = vec![0usize; arr.len()];
    for (i, &v) in arr.iter().enumerate() {
        args[pos[v]] = i;
        pos[v] += 1;
    }

    (args, locs)
}

fn crv1_meat_loop_imp(
    scores: &ArrayView2<f64>,
    clustid: &ArrayView1<usize>,
    cluster_col: &ArrayView1<usize>,
) -> Array2<f64> {
    let k = scores.ncols();
    let (g_indices, g_locs) = bucket_argsort_rs(cluster_col);

    // Compute cluster contributions
    let create_cluster_contrib = |&g: &usize| -> Array2<f64> {
        // Extract cluster indices
        let start = g_locs[g];
        let end = g_locs[g + 1];
        let col_indices = &g_indices[start..end];

        // Sum cluster scores
        let score_g = scores.select(Axis(0), col_indices).sum_axis(Axis(0));

        // Create the outer product
        let x = score_g.view().insert_axis(Axis(1));
        let x_t = score_g.view().insert_axis(Axis(0));
        x.dot(&x_t)
    };

    clustid
        .iter()
        .map(create_cluster_contrib)
        .fold(Array2::zeros((k, k)), |mut acc, x| {
            acc += &x;
            acc
        })
}

/// Compute the CRV1 meat matrix for cluster-robust standard errors.
///
/// Parameters
/// ----------
/// scores : numpy.ndarray (float64), shape (n_obs, k)
///     The score matrix, typically X' * u, where X is the design matrix and u are
///     the residuals from the model fit. Rows correspond to observations, columns to parameters.
/// clustid : numpy.ndarray (usize), shape (n_clusters,)
///     Array of unique cluster identifiers (one for each cluster).
/// cluster_col : numpy.ndarray (usize), shape (n_obs,)
///     Cluster assignment for each observation; each entry must match a value in `clustid`.
///
/// Returns
/// -------
/// meat : numpy.ndarray (float64), shape (k, k)
///     The CRV1 meat matrix (sum of cluster outer products), a square matrix where
///     k is the number of regression coefficients.
#[pyfunction]
pub fn _crv1_meat_loop_rs(
    py: Python,
    scores: PyReadonlyArray2<f64>,
    clustid: PyReadonlyArray1<usize>,
    cluster_col: PyReadonlyArray1<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    let meat = crv1_meat_loop_imp(
        &scores.as_array(),
        &clustid.as_array(),
        &cluster_col.as_array(),
    );
    Ok(meat.into_pyarray(py).to_owned().into())
}
