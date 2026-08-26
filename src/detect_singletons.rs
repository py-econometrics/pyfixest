use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Detect singleton fixed effects in a dataset.
///
/// This function iterates over the columns of a 2D numpy array representing
/// fixed effects to identify singleton fixed effects.
/// An observation is considered a singleton if it is the only one in its group
/// (fixed effect identifier).
///
/// # Arguments
/// * `ids` - A 2D numpy array of shape (n_samples, n_features) containing
///   non-negative integers representing fixed effect identifiers.
///
/// # Returns
/// A boolean array of shape (n_samples,), indicating which observations have
/// a singleton fixed effect.
///
/// # Notes
/// The algorithm iterates over columns to identify fixed effects. After each
/// column is processed, it updates the record of non-singleton rows. This approach
/// accounts for the possibility that removing an observation in one column can
/// lead to the emergence of new singletons in subsequent columns.
#[pyfunction]
pub fn _detect_singletons_rs(py: Python<'_>, ids: PyReadonlyArray2<u32>) -> Py<PyArray1<bool>> {
    let ids = ids.as_array();
    let (n_samples, n_features) = ids.dim();

    if n_samples == 0 {
        return vec![false; 0].into_pyarray(py).into();
    }

    // Find max value across all columns for count array sizing
    let max_fixef = ids.iter().cloned().max().unwrap_or(0) as usize;
    let mut counts = vec![0u32; max_fixef + 1];

    // Track non-singleton indices
    let mut non_singletons: Vec<u32> = (0..n_samples as u32).collect();
    let mut n_non_singletons = n_samples;

    loop {
        let n_non_singletons_curr = n_non_singletons;

        for j in 0..n_features {
            // Extract column once for faster 1D access (like numba does)
            let col = ids.column(j);

            // Reset counts
            counts.iter_mut().for_each(|c| *c = 0);

            // Count occurrences and track singleton count
            let mut n_singletons: i32 = 0;
            for i in 0..n_non_singletons {
                let idx = non_singletons[i] as usize;
                let e = col[idx] as usize;
                let c = counts[e];
                // Branchless version:
                // if c == 0: n_singletons += 1
                // if c == 1: n_singletons -= 1
                n_singletons += (c == 0) as i32 - (c == 1) as i32;
                counts[e] += 1;
            }

            if n_singletons == 0 {
                continue;
            }

            // Remove singletons from the non_singletons list
            let mut cnt = 0;
            for i in 0..n_non_singletons {
                let idx = non_singletons[i] as usize;
                let e = col[idx] as usize;
                if counts[e] != 1 {
                    non_singletons[cnt] = non_singletons[i];
                    cnt += 1;
                }
            }
            n_non_singletons = cnt;
        }

        if n_non_singletons_curr == n_non_singletons {
            break;
        }
    }

    // Build result: true means singleton
    let mut is_singleton = vec![true; n_samples];
    for i in 0..n_non_singletons {
        is_singleton[non_singletons[i] as usize] = false;
    }

    is_singleton.into_pyarray(py).into()
}
