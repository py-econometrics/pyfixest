use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

#[inline]
fn count_fixef_fully_nested(clusters: ArrayView1<usize>, f: ArrayView1<usize>) -> bool {
    let mut first_cluster: HashMap<usize, usize> = HashMap::new();
    for (&cl, &fv) in clusters.iter().zip(f.iter()) {
        match first_cluster.entry(fv) {
            Entry::Vacant(e) => {
                e.insert(cl);
            }
            Entry::Occupied(e) => {
                if *e.get() != cl {
                    return false;
                }
            }
        }
    }
    true
}

fn count_fixef_fully_nested_impl(
    all_fe: &[String],
    cluster_names: &[String],
    cdata: ArrayView2<usize>,
    fdata: ArrayView2<usize>,
) -> (Array1<bool>, usize) {
    let cluster_name_set: HashSet<&String> = cluster_names.iter().collect();

    // We allow
    let mut count = 0;
    let mask = all_fe
        .iter()
        .enumerate()
        .map(|(fi, fe_name)| {
            let is_nested = cluster_name_set.contains(fe_name) || {
                cdata
                    .axis_iter(Axis(1))
                    .any(|cluster_col| count_fixef_fully_nested(cluster_col, fdata.column(fi)))
            };
            if is_nested {
                count += 1
            }
            is_nested
        })
        .collect();

    (mask, count)
}


/// Compute which fixed effect columns are fully nested within any cluster variable,
/// and count the number of such columns.
///
/// Parameters
/// ----------
/// all_fixef_array : list of str
///     Names of all fixed effect variables in the model.
/// cluster_colnames : list of str
///     Names of all cluster variables in the model.
/// cluster_data : np.ndarray[usize]
///     2D array of cluster assignments (rows x cluster variables).
/// fe_data : np.ndarray[usize]
///     2D array of fixed effect values (rows x fixed effects).
///
/// Returns
/// -------
/// (np.ndarray[bool], int)
///     Tuple of (mask indicating which FEs are fully nested, count of such FEs).
///
/// Notes
/// -----
/// A fixed effect column is "fully nested" if for every unique value in that column,
/// all rows with that value share the same cluster assignment (for any cluster variable).

#[pyfunction]
pub fn _count_fixef_fully_nested_all_rs(
    all_fixef_array: &Bound<'_, PyAny>,
    cluster_colnames: &Bound<'_, PyAny>,
    cluster_data: PyReadonlyArray2<usize>,
    fe_data: PyReadonlyArray2<usize>,
) -> PyResult<(Py<PyArray1<bool>>, usize)> {
    // Get Python token from one of the bound parameters
    let py = all_fixef_array.py();

    // Extract Python data into Rust types
    let all_fe: Vec<String> = all_fixef_array.extract()?;
    let cluster_names: Vec<String> = cluster_colnames.extract()?;
    let cdata = cluster_data.as_array();
    let fdata = fe_data.as_array();

    // Call the pure Rust implementation
    let (mask, count) = count_fixef_fully_nested_impl(&all_fe, &cluster_names, cdata, fdata);

    // Convert back to Python objects
    let py_mask: Py<PyArray1<bool>> = mask.into_pyarray(py).to_owned().into();
    Ok((py_mask, count))
}
