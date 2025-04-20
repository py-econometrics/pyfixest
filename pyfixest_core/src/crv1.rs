use ndarray::{Array1, Array2};
use numpy::{IntoPyArray,
            PyArray2,      // for returning 2‑D arrays
            PyReadonlyArray1,  // <–– add this
            PyReadonlyArray2  // you already had this
};
use pyo3::prelude::*;

/// Pure Rust bucket‐argsort helper (not exposed to Python).
/// Takes a slice of cluster IDs, returns (indices, locs).
fn bucket_argsort_rs(arr: &[usize]) -> (Vec<usize>, Vec<usize>) {
    // 1) count frequencies
    let maxv = *arr.iter().max().unwrap_or(&0);
    let mut counts = vec![0; maxv + 1];
    for &v in arr { counts[v] += 1; }

    // 2) prefix sum → locs, init pos
    let m = counts.len();
    let mut locs = Vec::with_capacity(m+1);
    locs.push(0);
    let mut pos = Vec::with_capacity(m);
    for i in 0..m {
        pos.push(locs[i]);
        locs.push(locs[i] + counts[i]);
    }

    // 3) fill args
    let mut args = vec![0; arr.len()];
    for (i, &v) in arr.iter().enumerate() {
        let p = pos[v];
        args[p] = i;
        pos[v] += 1;
    }

    (args, locs)
}

/// The meat‐loop exposed to Python.
#[pyfunction]
pub fn crv1_meat_loop(
    py: Python,
    scores: PyReadonlyArray2<f64>,
    clustid: PyReadonlyArray1<usize>,
    cluster_col: PyReadonlyArray1<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    let sc = scores.as_array();
    let k = sc.ncols();

    // pull the cluster column into a Vec<usize>
    let cluster_vec: Vec<usize> = cluster_col.as_array().iter().cloned().collect();
    // call our private Rust helper
    let (g_indices, g_locs) = bucket_argsort_rs(&cluster_vec);

    // prepare accumulator
    let mut meat = Array2::<f64>::zeros((k,k));
    let mut meat_i = Array2::<f64>::zeros((k,k));

    // loop over each observation’s cluster ID
    let cl = clustid.as_array();
    for (&g, i) in cl.iter().zip(0..) {
        let start = g_locs[g];
        let end   = g_locs[g+1];
        let group = &g_indices[start..end];

        // sum the k‑vector for this group
        let mut score_g = Array1::<f64>::zeros(k);
        for &idx in group {
            for j in 0..k {
                score_g[j] += sc[[idx, j]];
            }
        }

        // outer product into meat_i
        for a in 0..k {
            for b in 0..k {
                meat_i[[a,b]] = score_g[a] * score_g[b];
            }
        }

        meat += &meat_i;
    }

    // hand back as a NumPy array
    Ok(meat.into_pyarray(py).to_owned())
}
