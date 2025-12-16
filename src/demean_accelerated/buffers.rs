//! Buffer management for acceleration algorithms.
//!
//! Provides single contiguous allocation for multiple working buffers,
//! reducing allocation overhead compared to separate Vec allocations.

/// A contiguous buffer that provides multiple logical slices.
///
/// Instead of allocating 6 separate `Vec<f64>` for acceleration buffers,
/// we allocate a single contiguous block and provide indexed access.
/// This improves cache locality and reduces allocation overhead.
///
/// # Example
/// ```ignore
/// let buf = CoefficientBuffer::new(1000, 6);
/// let x_curr = buf.buffer(0);      // First 1000 elements
/// let gx_curr = buf.buffer(1);     // Next 1000 elements
/// ```
pub struct CoefficientBuffer {
    data: Vec<f64>,
    n_samples: usize,
    n_buffers: usize,
}

impl CoefficientBuffer {
    /// Create a new buffer with `n_buffers` logical slices of `n_samples` each.
    ///
    /// Total allocation: `n_samples * n_buffers` f64 values.
    pub fn new(n_samples: usize, n_buffers: usize) -> Self {
        Self {
            data: vec![0.0; n_samples * n_buffers],
            n_samples,
            n_buffers,
        }
    }

    /// Get the number of samples (elements per buffer).
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get the number of logical buffers.
    #[inline]
    pub fn n_buffers(&self) -> usize {
        self.n_buffers
    }

    /// Get an immutable slice for buffer at index `idx`.
    ///
    /// # Panics
    /// Panics if `idx >= n_buffers`.
    #[inline]
    pub fn buffer(&self, idx: usize) -> &[f64] {
        debug_assert!(idx < self.n_buffers, "buffer index out of bounds");
        let start = idx * self.n_samples;
        let end = start + self.n_samples;
        &self.data[start..end]
    }

    /// Get a mutable slice for buffer at index `idx`.
    ///
    /// # Panics
    /// Panics if `idx >= n_buffers`.
    #[inline]
    pub fn buffer_mut(&mut self, idx: usize) -> &mut [f64] {
        debug_assert!(idx < self.n_buffers, "buffer index out of bounds");
        let start = idx * self.n_samples;
        let end = start + self.n_samples;
        &mut self.data[start..end]
    }

    /// Fill all buffers with zeros.
    #[inline]
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    /// Copy data from one buffer to another within the same CoefficientBuffer.
    ///
    /// # Panics
    /// Panics if `src_idx` or `dst_idx >= n_buffers`, or if `src_idx == dst_idx`.
    pub fn copy_buffer(&mut self, src_idx: usize, dst_idx: usize) {
        debug_assert!(src_idx < self.n_buffers, "source buffer index out of bounds");
        debug_assert!(dst_idx < self.n_buffers, "destination buffer index out of bounds");
        debug_assert!(src_idx != dst_idx, "source and destination cannot be the same");

        let src_start = src_idx * self.n_samples;
        let dst_start = dst_idx * self.n_samples;

        // Use copy_within for efficient in-place copy
        self.data
            .copy_within(src_start..src_start + self.n_samples, dst_start);
    }

    /// Get one immutable and one mutable buffer simultaneously.
    ///
    /// This uses split_at_mut to safely provide both references without copying.
    ///
    /// # Panics
    /// Panics if indices are out of bounds or equal.
    #[inline]
    pub fn get_read_write(&mut self, read_idx: usize, write_idx: usize) -> (&[f64], &mut [f64]) {
        debug_assert!(read_idx < self.n_buffers, "read index out of bounds");
        debug_assert!(write_idx < self.n_buffers, "write index out of bounds");
        debug_assert_ne!(read_idx, write_idx, "indices must be different");

        let n = self.n_samples;
        if read_idx < write_idx {
            let (left, right) = self.data.split_at_mut(write_idx * n);
            (&left[read_idx * n..(read_idx + 1) * n], &mut right[..n])
        } else {
            let (left, right) = self.data.split_at_mut(read_idx * n);
            (&right[..n], &mut left[write_idx * n..(write_idx + 1) * n])
        }
    }

    /// Get three immutable buffers and one mutable buffer simultaneously.
    ///
    /// # Panics
    /// Panics if any indices are out of bounds or if write_idx equals any read index.
    #[inline]
    pub fn get_3read_1write(
        &mut self,
        r0: usize,
        r1: usize,
        r2: usize,
        w: usize,
    ) -> (&[f64], &[f64], &[f64], &mut [f64]) {
        debug_assert!(r0 < self.n_buffers && r1 < self.n_buffers && r2 < self.n_buffers);
        debug_assert!(w < self.n_buffers);
        debug_assert!(w != r0 && w != r1 && w != r2, "write index must differ from read indices");

        let n = self.n_samples;

        // Find the split point - we split just before the write buffer
        let w_start = w * n;
        let (left, right) = self.data.split_at_mut(w_start);
        let (write_buf, after_write) = right.split_at_mut(n);

        // Helper to get a read slice given the split
        let get_read = |idx: usize| -> &[f64] {
            let start = idx * n;
            if start < w_start {
                &left[start..start + n]
            } else {
                let offset = start - w_start - n;
                &after_write[offset..offset + n]
            }
        };

        (get_read(r0), get_read(r1), get_read(r2), write_buf)
    }

    /// Get direct mutable access to underlying data for complex operations.
    ///
    /// Returns the data slice and n_samples for manual indexing.
    #[inline]
    pub fn raw_data_mut(&mut self) -> (&mut [f64], usize) {
        (&mut self.data, self.n_samples)
    }
}

/// Buffer indices for Irons-Tuck acceleration.
///
/// Provides named constants for buffer indices, making code more readable.
pub mod indices {
    /// Current iterate X
    pub const X_CURR: usize = 0;
    /// G(X) - first projection
    pub const GX_CURR: usize = 1;
    /// G(G(X)) - second projection
    pub const GGX_CURR: usize = 2;
    /// GGX - GX (working buffer for delta)
    pub const DELTA_GX: usize = 3;
    /// Delta2_X (acceleration calculation buffer)
    pub const DELTA2_X: usize = 4;
    /// Previous X (for convergence checking)
    pub const X_PREV: usize = 5;

    /// Number of buffers needed for Irons-Tuck acceleration
    pub const IRONS_TUCK_BUFFER_COUNT: usize = 6;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_buffer() {
        let buf = CoefficientBuffer::new(100, 6);
        assert_eq!(buf.n_samples(), 100);
        assert_eq!(buf.n_buffers(), 6);
        assert_eq!(buf.data.len(), 600);
    }

    #[test]
    fn test_buffer_access() {
        let mut buf = CoefficientBuffer::new(10, 3);

        // Write to first buffer
        buf.buffer_mut(0)[0] = 1.0;
        buf.buffer_mut(0)[9] = 10.0;

        // Write to second buffer
        buf.buffer_mut(1)[0] = 100.0;

        // Write to third buffer
        buf.buffer_mut(2)[5] = 500.0;

        // Read back
        assert_eq!(buf.buffer(0)[0], 1.0);
        assert_eq!(buf.buffer(0)[9], 10.0);
        assert_eq!(buf.buffer(1)[0], 100.0);
        assert_eq!(buf.buffer(2)[5], 500.0);

        // Verify no overlap
        assert_eq!(buf.buffer(1)[9], 0.0);
        assert_eq!(buf.buffer(2)[0], 0.0);
    }

    #[test]
    fn test_clear() {
        let mut buf = CoefficientBuffer::new(10, 2);
        buf.buffer_mut(0)[0] = 1.0;
        buf.buffer_mut(1)[5] = 2.0;

        buf.clear();

        assert_eq!(buf.buffer(0)[0], 0.0);
        assert_eq!(buf.buffer(1)[5], 0.0);
    }

    #[test]
    fn test_copy_buffer() {
        let mut buf = CoefficientBuffer::new(5, 3);

        // Set up source buffer
        for (i, v) in buf.buffer_mut(0).iter_mut().enumerate() {
            *v = i as f64;
        }

        // Copy to destination
        buf.copy_buffer(0, 2);

        // Verify copy
        assert_eq!(buf.buffer(2), buf.buffer(0));
        assert_eq!(buf.buffer(2)[0], 0.0);
        assert_eq!(buf.buffer(2)[4], 4.0);
    }

    #[test]
    fn test_empty_buffer() {
        let buf = CoefficientBuffer::new(0, 3);
        assert_eq!(buf.n_samples(), 0);
        assert_eq!(buf.buffer(0).len(), 0);
        assert_eq!(buf.buffer(1).len(), 0);
    }

    #[test]
    fn test_indices() {
        use indices::*;
        assert_eq!(X_CURR, 0);
        assert_eq!(GX_CURR, 1);
        assert_eq!(GGX_CURR, 2);
        assert_eq!(DELTA_GX, 3);
        assert_eq!(DELTA2_X, 4);
        assert_eq!(X_PREV, 5);
        assert_eq!(IRONS_TUCK_BUFFER_COUNT, 6);
    }
}
