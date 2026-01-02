//! Working buffer management for demeaning operations.
//!
//! This module re-exports buffer types from their primary locations.
//! Most code should import directly from [`acceleration`] or [`projection`].

// Re-exports for backward compatibility and convenience
#[allow(unused_imports)]
pub use crate::demean_accelerated::acceleration::AccelBuffers;
#[allow(unused_imports)]
pub use crate::demean_accelerated::projection::{MultiFEScratch, TwoFEScratch};
