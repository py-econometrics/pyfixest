use ndarray::{Array1, Array2};
use numpy::{IntoPyArray,
            PyArray2,
            PyReadonlyArray1,
            PyReadonlyArray2
};
use pyo3::prelude::*;

fn bucket_argsort_rs(arr: &[usize]) -> (Vec<usize>, Vec<usize>) {
    // 1) count frequencies
    let maxv = *arr.iter().max().unwrap_or(&0);
    let mut counts = vec![0; maxv + 1];
    for &v in arr { counts[v] += 1; }

    let m = counts.len();
    let mut locs = Vec::with_capacity(m+1);
    locs.push(0);
    let mut pos = Vec::with_capacity(m);
    for i in 0..m {
        pos.push(locs[i]);
        locs.push(locs[i] + counts[i]);
    }

    let mut args = vec![0; arr.len()];
    for (i, &v) in arr.iter().enumerate() {
        let p = pos[v];
        args[p] = i;
        pos[v] += 1;
    }

    (args, locs)
}

#[pyfunction]
pub fn crv1_meat_loop_rs(
    py: Python,
    scores: PyReadonlyArray2<f64>,
    clustid: PyReadonlyArray1<usize>,
    cluster_col: PyReadonlyArray1<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    let sc = scores.as_array();
    let k = sc.ncols();

    let cluster_vec: Vec<usize> = cluster_col.as_array().iter().cloned().collect();
    let (g_indices, g_locs) = bucket_argsort_rs(&cluster_vec);

    let mut meat = Array2::<f64>::zeros((k,k));
    let mut meat_i = Array2::<f64>::zeros((k,k));

    let cl = clustid.as_array();
    for (&g, _i) in cl.iter().zip(0..) {
        let start = g_locs[g];
        let end   = g_locs[g+1];
        let group = &g_indices[start..end];

        let mut score_g = Array1::<f64>::zeros(k);
        for &idx in group {
            for j in 0..k {
                score_g[j] += sc[[idx, j]];
            }
        }

        for a in 0..k {
            for b in 0..k {
                meat_i[[a,b]] = score_g[a] * score_g[b];
            }
        }

        meat += &meat_i;
    }

    Ok(meat.into_pyarray(py).to_owned())
}
