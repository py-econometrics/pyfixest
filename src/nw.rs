use ndarray::{Array1, Array2, ArrayView2, s};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute Bartlett kernel weights for a given lag.
fn bartlett_weights(lag: usize) -> Array1<f64> {
    let lag_plus_one = (lag + 1) as f64;
    let mut weights = Array1::zeros(lag + 1);
    for j in 0..=lag {
        weights[j] = 1.0 - j as f64 / lag_plus_one;
    }
    weights[0] = 0.5; // halve first weight (diagonal counted once)
    weights
}

/// Core HAC meat loop shared by time-series NW and DK.
///
/// `scores` is (T, k), already sorted by time.
fn hac_meat_loop(scores: &ArrayView2<f64>, lag: usize) -> Array2<f64> {
    let time_periods = scores.nrows();
    let k = scores.ncols();
    let weights = bartlett_weights(lag);

    let mut meat = Array2::<f64>::zeros((k, k));
    let mut gamma = Array2::<f64>::zeros((k, k));

    for l in 0..=lag.min(time_periods.saturating_sub(1)) {
        let current = scores.slice(s![l..time_periods, ..]);
        let lagged = scores.slice(s![0..time_periods - l, ..]);
        gamma.assign(&current.t().dot(&lagged));

        let w = weights[l];
        meat.scaled_add(w, &gamma);
        meat.scaled_add(w, &gamma.t());
    }

    meat
}

/// Panel Newey-West HAC meat matrix.
///
/// `scores` is (N_total, k), sorted by (panel, time).
/// `starts[i]` / `counts[i]` define per-panel slices.
fn nw_meat_panel_impl(
    scores: &ArrayView2<f64>,
    starts: &[usize],
    counts: &[usize],
    lag: usize,
) -> Array2<f64> {
    let k = scores.ncols();
    let weights = bartlett_weights(lag);

    let mut meat = Array2::<f64>::zeros((k, k));
    let mut gamma_l_sum = Array2::<f64>::zeros((k, k));

    for (&start, &count) in starts.iter().zip(counts.iter()) {
        let end = start + count;
        let score_i = scores.slice(s![start..end, ..]);

        // gamma_0 = score_i^T @ score_i
        let gamma0 = score_i.t().dot(&score_i);

        gamma_l_sum.fill(0.0);
        let l_max = lag.min(count.saturating_sub(1));

        for l in 1..=l_max {
            let score_curr = scores.slice(s![(start + l)..end, ..]);
            let score_prev = scores.slice(s![start..(end - l), ..]);
            let gamma_l = score_curr.t().dot(&score_prev);
            let w = weights[l];
            gamma_l_sum.scaled_add(w, &gamma_l);
            gamma_l_sum.scaled_add(w, &gamma_l.t());
        }

        meat += &gamma0;
        meat += &gamma_l_sum;
    }

    meat
}

/// Time-series Newey-West HAC meat. Sorts scores by `time_arr`, then
/// delegates to `hac_meat_loop`.
fn nw_meat_time_impl(scores: &ArrayView2<f64>, time_arr: &[f64], lag: usize) -> Array2<f64> {
    // argsort by time
    let mut order: Vec<usize> = (0..time_arr.len()).collect();
    order.sort_by(|&a, &b| time_arr[a].partial_cmp(&time_arr[b]).unwrap());

    let sorted_scores = scores.select(ndarray::Axis(0), &order);
    hac_meat_loop(&sorted_scores.view(), lag)
}

/// Driscoll-Kraay HAC meat. Aggregates scores by time period, then
/// delegates to `hac_meat_loop`.
///
/// `scores` is (N_total, k), sorted by time.
/// `idx[t]` is the start index of time period t.
fn dk_meat_panel_impl(
    scores: &ArrayView2<f64>,
    idx: &[usize],
    lag: usize,
) -> Array2<f64> {
    let n_times = idx.len();
    let k = scores.ncols();
    let n_obs = scores.nrows();

    let mut scores_time = Array2::<f64>::zeros((n_times, k));
    for t in 0..n_times {
        let start = idx[t];
        let end = if t + 1 < n_times { idx[t + 1] } else { n_obs };
        let slice = scores.slice(s![start..end, ..]);
        scores_time
            .row_mut(t)
            .assign(&slice.sum_axis(ndarray::Axis(0)));
    }

    hac_meat_loop(&scores_time.view(), lag)
}

// ---------------------------------------------------------------------------
// PyO3 wrappers
// ---------------------------------------------------------------------------

/// Compute the panel Newey-West (HAC) meat matrix.
///
/// Parameters
/// ----------
/// scores : ndarray (float64), shape (n_obs, k)
///     Score matrix sorted by (panel, time).
/// starts : ndarray (uint64), shape (n_panels,)
///     Start index of each panel unit.
/// counts : ndarray (uint64), shape (n_panels,)
///     Number of observations per panel unit.
/// lag : int
///     Maximum lag for autocovariance (Bartlett kernel bandwidth).
///
/// Returns
/// -------
/// meat : ndarray (float64), shape (k, k)
#[pyfunction]
pub fn _nw_meat_panel_rs(
    py: Python,
    scores: PyReadonlyArray2<f64>,
    starts: PyReadonlyArray1<usize>,
    counts: PyReadonlyArray1<usize>,
    lag: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let scores_view = scores.as_array();
    let starts_slice = starts.as_slice()?;
    let counts_slice = counts.as_slice()?;
    let meat = nw_meat_panel_impl(&scores_view, starts_slice, counts_slice, lag);
    Ok(meat.into_pyarray(py).to_owned().into())
}

/// Compute the time-series Newey-West (HAC) meat matrix.
///
/// Parameters
/// ----------
/// scores : ndarray (float64), shape (T, k)
///     Score matrix (may be unsorted).
/// time_arr : ndarray (float64), shape (T,)
///     Time variable used to sort scores.
/// lag : int
///     Maximum lag for autocovariance.
///
/// Returns
/// -------
/// meat : ndarray (float64), shape (k, k)
#[pyfunction]
pub fn _nw_meat_time_rs(
    py: Python,
    scores: PyReadonlyArray2<f64>,
    time_arr: PyReadonlyArray1<f64>,
    lag: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let scores_view = scores.as_array();
    let time_slice = time_arr.as_slice()?;
    let meat = nw_meat_time_impl(&scores_view, time_slice, lag);
    Ok(meat.into_pyarray(py).to_owned().into())
}

/// Compute the Driscoll-Kraay (HAC) meat matrix.
///
/// Parameters
/// ----------
/// scores : ndarray (float64), shape (n_obs, k)
///     Score matrix sorted by time.
/// idx : ndarray (uint64), shape (n_times,)
///     Start index of each time period in the sorted scores.
/// lag : int
///     Maximum lag for autocovariance.
///
/// Returns
/// -------
/// meat : ndarray (float64), shape (k, k)
#[pyfunction]
pub fn _dk_meat_panel_rs(
    py: Python,
    scores: PyReadonlyArray2<f64>,
    idx: PyReadonlyArray1<usize>,
    lag: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let scores_view = scores.as_array();
    let idx_slice = idx.as_slice()?;
    let meat = dk_meat_panel_impl(&scores_view, idx_slice, lag);
    Ok(meat.into_pyarray(py).to_owned().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_bartlett_weights() {
        let w = bartlett_weights(4);
        assert_eq!(w.len(), 5);
        assert!((w[0] - 0.5).abs() < 1e-15);
        assert!((w[1] - 0.8).abs() < 1e-15);
        assert!((w[2] - 0.6).abs() < 1e-15);
        assert!((w[3] - 0.4).abs() < 1e-15);
        assert!((w[4] - 0.2).abs() < 1e-15);
    }

    #[test]
    fn test_hac_meat_loop_lag_zero() {
        // lag=0 → meat = 0.5*(S'S + S'S) = S'S
        let scores = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let meat = hac_meat_loop(&scores.view(), 0);
        let expected = scores.t().dot(&scores);
        assert!((meat - expected).mapv(f64::abs).sum() < 1e-10);
    }

    #[test]
    fn test_nw_panel_lag_zero() {
        // lag=0 → meat = S'S (sum over panels)
        let scores = array![[1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
        let starts = vec![0usize, 2];
        let counts = vec![2usize, 2];
        let meat = nw_meat_panel_impl(&scores.view(), &starts, &counts, 0);
        let expected = scores.t().dot(&scores);
        assert!((meat - expected).mapv(f64::abs).sum() < 1e-10);
    }

    #[test]
    fn test_nw_meat_time_sorts() {
        // Unsorted input should produce same result as sorted
        let scores = array![[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]];
        let time_arr = vec![3.0, 1.0, 2.0];
        let meat = nw_meat_time_impl(&scores.view(), &time_arr, 0);

        let sorted = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let expected = hac_meat_loop(&sorted.view(), 0);
        assert!((meat - expected).mapv(f64::abs).sum() < 1e-10);
    }

    #[test]
    fn test_dk_aggregation() {
        // 2 obs per time, 3 times
        let scores = array![
            [1.0, 2.0], [3.0, 4.0],
            [5.0, 6.0], [7.0, 8.0],
            [9.0, 10.0], [11.0, 12.0]
        ];
        let idx = vec![0usize, 2, 4];
        let meat = dk_meat_panel_impl(&scores.view(), &idx, 0);

        // Aggregated: [4,6], [12,14], [20,22]
        let agg = array![[4.0, 6.0], [12.0, 14.0], [20.0, 22.0]];
        let expected = agg.t().dot(&agg);
        assert!((meat - expected).mapv(f64::abs).sum() < 1e-10);
    }
}
