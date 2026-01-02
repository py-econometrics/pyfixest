//! Working buffer management for demeaning operations.

/// Working buffers for the acceleration loop.
///
/// Contains only the acceleration state vectors. Projection scratch
/// is owned by individual projectors.
pub struct DemeanBuffers {
    /// G(x): Result of one projection step.
    pub gx: Vec<f64>,
    /// G(G(x)): Result of two projection steps.
    pub ggx: Vec<f64>,
    /// Temporary buffer for post-acceleration projection.
    pub temp: Vec<f64>,
    /// Grand acceleration: y snapshot.
    pub y: Vec<f64>,
    /// Grand acceleration: G(y) snapshot.
    pub gy: Vec<f64>,
    /// Grand acceleration: G(G(y)) snapshot.
    pub ggy: Vec<f64>,
}

impl DemeanBuffers {
    /// Create buffers for demeaning.
    ///
    /// # Arguments
    /// * `n_coef` - Total number of coefficients (sum of groups across all FEs)
    pub fn new(n_coef: usize) -> Self {
        Self {
            gx: vec![0.0; n_coef],
            ggx: vec![0.0; n_coef],
            temp: vec![0.0; n_coef],
            y: vec![0.0; n_coef],
            gy: vec![0.0; n_coef],
            ggy: vec![0.0; n_coef],
        }
    }
}
