use ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::f64::consts::PI;

fn py_value_error(message: &str) -> PyErr {
    PyValueError::new_err(message.to_string())
}

#[inline]
fn fabs_lon(lon_rad_1: f64, lon_rad_2: f64) -> f64 {
    let diff = (lon_rad_1 - lon_rad_2).abs();
    let diff = if diff >= 2.0 * PI {
        diff.rem_euclid(2.0 * PI)
    } else {
        diff
    };
    diff.min(2.0 * PI - diff)
}

fn conley_meat_impl(
    scores: &ArrayView2<f64>,
    lon_arr: &ArrayView1<f64>,
    lat_arr: &ArrayView1<f64>,
    distance: usize,
    cutoff: f64,
) -> PyResult<Array2<f64>> {
    if distance == 0 || distance > 2 {
        return Err(py_value_error("'distance' is not valid."));
    }

    if !cutoff.is_finite() || cutoff < 0.0 {
        return Err(py_value_error(
            "The Conley cutoff must be non-negative and finite.",
        ));
    }

    let n = scores.nrows();
    let k = scores.ncols();

    if lon_arr.len() != n || lat_arr.len() != n {
        return Err(py_value_error(
            "Scores, longitude, and latitude arrays must have the same number of observations.",
        ));
    }

    let lon_rad: Vec<f64> = lon_arr.iter().map(|&x| x.to_radians()).collect();
    let lat_rad: Vec<f64> = lat_arr.iter().map(|&x| x.to_radians()).collect();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        lat_rad[a]
            .partial_cmp(&lat_rad[b])
            .unwrap_or(Ordering::Equal)
    });

    let mut scores_sorted = Array2::<f64>::zeros((n, k));
    let mut lon_sorted = vec![0.0; n];
    let mut lat_sorted = vec![0.0; n];
    for (i, &idx) in order.iter().enumerate() {
        scores_sorted.row_mut(i).assign(&scores.row(idx));
        lon_sorted[i] = lon_rad[idx];
        lat_sorted[i] = lat_rad[idx];
    }

    let mut cos_lat = vec![0.0; n];
    for i in 0..n {
        cos_lat[i] = lat_sorted[i].cos();
    }

    let lat_cutoff_rad = (cutoff / 111.0).to_radians();
    let lon_cutoff_rad_factor = (cutoff / 111.0).to_radians();
    let cutoff_rad_sq = lat_cutoff_rad * lat_cutoff_rad;
    let earth_diameter_km = 12752.0;

    let mut lon_cutoff = vec![0.0; n];
    for i in 0..n {
        let cos_lat_abs = cos_lat[i].abs();
        lon_cutoff[i] = if cos_lat_abs < 1e-15 {
            f64::INFINITY
        } else {
            lon_cutoff_rad_factor / cos_lat_abs
        };
    }

    let mut cum_scores = scores_sorted.clone();
    cum_scores *= 0.5;

    for i in 0..n {
        let lon_rad_i = lon_sorted[i];
        let lat_rad_i = lat_sorted[i];
        let cos_lat_i = cos_lat[i];

        for j in (i + 1)..n {
            let dist_lat_rad = lat_sorted[j] - lat_rad_i;
            if dist_lat_rad > lat_cutoff_rad {
                break;
            }

            let dist_lon_rad = fabs_lon(lon_sorted[j], lon_rad_i);

            let cos_lat_mean = ((lat_rad_i + lat_sorted[j]) / 2.0).cos();

            if distance == 2 {
                // Fast planar pruning is valid only for the triangular distance.
                let max_lon_cutoff = lon_cutoff[i].max(lon_cutoff[j]);
                if dist_lon_rad > max_lon_cutoff {
                    continue;
                }
            }

            let cos_lat_mean_abs = cos_lat_mean.abs();
            let lon_cutoff_rad = if cos_lat_mean_abs < 1e-15 {
                f64::INFINITY
            } else {
                lon_cutoff_rad_factor / cos_lat_mean_abs
            };

            if distance == 2 && dist_lon_rad > lon_cutoff_rad {
                continue;
            }

            let ok = if distance == 1 {
                let sin_dlat = (dist_lat_rad / 2.0).sin();
                let sin_dlon = (dist_lon_rad / 2.0).sin();
                let mut a = sin_dlat * sin_dlat + cos_lat_i * cos_lat[j] * sin_dlon * sin_dlon;
                if a > 1.0 {
                    a = 1.0;
                }
                let dist = earth_diameter_km * a.sqrt().asin();
                dist <= cutoff
            } else {
                let scaled_lon = cos_lat_mean * dist_lon_rad;
                let dist = dist_lat_rad * dist_lat_rad + scaled_lon * scaled_lon;
                dist <= cutoff_rad_sq
            };

            if ok {
                for col in 0..k {
                    cum_scores[[i, col]] += scores_sorted[[j, col]];
                }
            }
        }
    }

    // Highly optimized BLAS / matrix multiplication replacement for nested loops
    let mut res = scores_sorted.t().dot(&cum_scores);

    for k1 in 0..k {
        for k2 in k1..k {
            if k1 == k2 {
                res[[k1, k2]] *= 2.0;
            } else {
                let tmp = res[[k1, k2]];
                res[[k1, k2]] += res[[k2, k1]];
                res[[k2, k1]] += tmp;
            }
        }
    }

    Ok(res)
}

/// Compute the Conley spatial HAC meat matrix.
///
/// Parameters
/// ----------
/// scores : ndarray (float64), shape (n_obs, k)
///     Score matrix.
/// lon_arr : ndarray (float64), shape (n_obs,)
///     Normalized longitude in degrees.
/// lat_arr : ndarray (float64), shape (n_obs,)
///     Normalized latitude in degrees.
/// distance : int
///     Distance mode: 1 for spherical, 2 for triangular.
/// cutoff : float
///     Cutoff distance in kilometers.
///
/// Returns
/// -------
/// meat : ndarray (float64), shape (k, k)
#[pyfunction]
pub fn _conley_meat_rs(
    py: Python,
    scores: PyReadonlyArray2<f64>,
    lon_arr: PyReadonlyArray1<f64>,
    lat_arr: PyReadonlyArray1<f64>,
    distance: usize,
    cutoff: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let scores_view = scores.as_array();
    let lon_view = lon_arr.as_array();
    let lat_view = lat_arr.as_array();

    // Release the Python GIL to allow concurrent threads during heavy computation.
    let meat =
        py.detach(|| conley_meat_impl(&scores_view, &lon_view, &lat_view, distance, cutoff))?;

    Ok(meat.into_pyarray(py).to_owned().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn assert_all_close(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64) {
        assert_eq!(actual.shape(), expected.shape());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= tol, "actual: {a}, expected: {e}");
        }
    }

    #[test]
    fn test_tiny_cutoff_no_neighbors() {
        let scores = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let lon = vec![0.0, 20.0, 40.0];
        let lat = vec![0.0, 20.0, 40.0];
        let lon_view = ArrayView1::from(&lon);
        let lat_view = ArrayView1::from(&lat);
        let meat = conley_meat_impl(&scores.view(), &lon_view, &lat_view, 2, 1.0).unwrap();
        let expected = scores.t().dot(&scores);
        assert_all_close(&meat, &expected, 1e-12);
    }

    #[test]
    fn test_huge_cutoff_all_pairs() {
        let scores = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let lon = vec![0.0, 20.0, 40.0];
        let lat = vec![0.0, 20.0, 40.0];
        let lon_view = ArrayView1::from(&lon);
        let lat_view = ArrayView1::from(&lat);
        let meat = conley_meat_impl(&scores.view(), &lon_view, &lat_view, 2, 20_000.0).unwrap();
        let score_sum = scores.sum_axis(ndarray::Axis(0));
        let expected = score_sum
            .view()
            .insert_axis(ndarray::Axis(1))
            .dot(&score_sum.view().insert_axis(ndarray::Axis(0)));
        assert_all_close(&meat, &expected, 1e-12);
    }

    #[test]
    fn test_spherical_high_latitude_neighbor() {
        let scores = array![[1.0], [2.0]];
        let lon = vec![0.0, 120.0];
        let lat = vec![80.0, 80.0];
        let lon_view = ArrayView1::from(&lon);
        let lat_view = ArrayView1::from(&lat);
        let meat = conley_meat_impl(&scores.view(), &lon_view, &lat_view, 1, 2_000.0).unwrap();
        let expected = array![[9.0]];
        assert_all_close(&meat, &expected, 1e-12);
    }

    #[test]
    fn test_latitude_uses_normalized_degrees() {
        let scores = array![[1.0], [2.0]];
        let lon = vec![0.0, 1.0];
        let lat = vec![10.0, 10.0];
        let lon_view = ArrayView1::from(&lon);
        let lat_view = ArrayView1::from(&lat);

        let meat = conley_meat_impl(&scores.view(), &lon_view, &lat_view, 2, 50.0).unwrap();
        let expected = array![[9.0]];
        assert_all_close(&meat, &expected, 1e-12);
    }
}
