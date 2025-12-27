//! Working buffer management for demeaning operations.
//!
//! Pre-allocated buffers to reduce allocation overhead during iteration.

/// Working buffers for 2-FE acceleration loop.
pub struct TwoFEBuffers {
    /// G(x) buffer (size: n_groups[0])
    pub gx: Vec<f64>,
    /// G(G(x)) buffer (size: n_groups[0])
    pub ggx: Vec<f64>,
    /// Temporary buffer (size: n_groups[0])
    pub temp: Vec<f64>,
    /// Beta temporary buffer (size: n_groups[1])
    pub beta_tmp: Vec<f64>,
    /// Grand acceleration: y (size: n_groups[0])
    pub y: Vec<f64>,
    /// Grand acceleration: G(y) (size: n_groups[0])
    pub gy: Vec<f64>,
    /// Grand acceleration: G(G(y)) (size: n_groups[0])
    pub ggy: Vec<f64>,
}

impl TwoFEBuffers {
    /// Create new buffers for 2-FE acceleration.
    pub fn new(n0: usize, n1: usize) -> Self {
        Self {
            gx: vec![0.0; n0],
            ggx: vec![0.0; n0],
            temp: vec![0.0; n0],
            beta_tmp: vec![0.0; n1],
            y: vec![0.0; n0],
            gy: vec![0.0; n0],
            ggy: vec![0.0; n0],
        }
    }
}

/// Working buffers for Q-FE (3+) acceleration loop.
pub struct MultiFEBuffers {
    /// G(x) buffer (size: n_coef_total)
    pub gx: Vec<f64>,
    /// G(G(x)) buffer (size: n_coef_total)
    pub ggx: Vec<f64>,
    /// Temporary buffer (size: n_coef_total)
    pub temp: Vec<f64>,
    /// Sum of other FE means buffer (size: n_obs)
    pub sum_other_means: Vec<f64>,
    /// Grand acceleration: y (size: n_coef_total)
    pub y: Vec<f64>,
    /// Grand acceleration: G(y) (size: n_coef_total)
    pub gy: Vec<f64>,
    /// Grand acceleration: G(G(y)) (size: n_coef_total)
    pub ggy: Vec<f64>,
    /// Output buffer for SSR computation (size: n_obs)
    pub output_buf: Vec<f64>,
}

impl MultiFEBuffers {
    /// Create new buffers for Q-FE acceleration.
    pub fn new(n_coef: usize, n_obs: usize) -> Self {
        Self {
            gx: vec![0.0; n_coef],
            ggx: vec![0.0; n_coef],
            temp: vec![0.0; n_coef],
            sum_other_means: vec![0.0; n_obs],
            y: vec![0.0; n_coef],
            gy: vec![0.0; n_coef],
            ggy: vec![0.0; n_coef],
            output_buf: vec![0.0; n_obs],
        }
    }
}
