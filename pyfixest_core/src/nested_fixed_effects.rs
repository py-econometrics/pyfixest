use std::collections::HashMap;
use ndarray::{Array1, ArrayView1};
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray2, IntoPyArray};

fn count_fixef_fully_nested(
    clusters: ArrayView1<usize>,
    f: ArrayView1<usize>,
) -> bool {
    let mut first_cluster: HashMap<usize, usize> = HashMap::new();
    for (&cl, &fv) in clusters.iter().zip(f.iter()) {
        use std::collections::hash_map::Entry;
        match first_cluster.entry(fv) {
            Entry::Vacant(e) => { e.insert(cl); }
            Entry::Occupied(mut e) => {
                if *e.get() != cl {
                    e.insert(usize::MAX);
                }
            }
        }
    }
    first_cluster.values().all(|&c| c != usize::MAX)
}

#[pyfunction]
pub fn count_fixef_fully_nested_all_rs(
    py: Python<'_>,
    all_fixef_array: &PyAny,
    cluster_colnames: &PyAny,
    cluster_data: PyReadonlyArray2<usize>,
    fe_data:       PyReadonlyArray2<usize>,
) -> PyResult<(Py<PyArray1<bool>>, usize)> {
    let all_fe: Vec<String>      = all_fixef_array.extract()?;
    let cluster_names: Vec<String> = cluster_colnames.extract()?;

    let cdata = cluster_data.as_array();
    let fdata = fe_data.as_array();

    let n_feat = all_fe.len();
    let mut mask  = Array1::from_elem(n_feat, false);
    let mut count = 0;

    for fi in 0..n_feat {
        let fe_name = &all_fe[fi];
        if cluster_names.iter().any(|c| c == fe_name) {
            mask[fi] = true;
            count += 1;
            continue;
        }
        for col_j in 0..cdata.ncols() {
            let clusters_col = cdata.column(col_j);
            let fe_col       = fdata.column(fi);
            if count_fixef_fully_nested(clusters_col, fe_col) {
                mask[fi] = true;
                count += 1;
                break;
            }
        }
    }

    let py_mask: Py<PyArray1<bool>> = mask.into_pyarray(py).to_owned();
    Ok((py_mask, count))
}

#[pymodule]
fn your_module(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_fixef_fully_nested_all_rs, m)?)?;
    Ok(())
}
