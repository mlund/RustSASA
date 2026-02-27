// Copyright (c) 2024 Maxwell Campbell. Licensed under the MIT License.

/// Trait for distance calculations with different boundary conditions.
///
/// This trait is used to abstract over Euclidean (non-periodic) and
/// periodic boundary condition distance calculations. By using generics
/// with this trait, the compiler generates separate optimized code paths
/// for each implementation, ensuring zero overhead for the non-periodic case.
pub trait DistanceMetric: Copy + Send + Sync {
    /// Whether this metric uses periodic boundary conditions.
    ///
    /// Used to determine if cell coordinates should wrap around grid boundaries.
    const IS_PERIODIC: bool;

    /// Calculate the displacement component along one dimension.
    ///
    /// For Euclidean: simply `a - b`
    /// For Periodic: applies minimum image convention
    fn delta(&self, a: f32, b: f32, dim: usize) -> f32;

    /// Calculate displacement vector from point `a` to point `b`.
    #[inline(always)]
    fn displacement(&self, a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
        [
            self.delta(a[0], b[0], 0),
            self.delta(a[1], b[1], 1),
            self.delta(a[2], b[2], 2),
        ]
    }

    /// Calculate squared distance between two points.
    #[inline(always)]
    fn distance_squared(&self, a: &[f32; 3], b: &[f32; 3]) -> f32 {
        let dx = self.delta(a[0], b[0], 0);
        let dy = self.delta(a[1], b[1], 1);
        let dz = self.delta(a[2], b[2], 2);
        dx * dx + dy * dy + dz * dz
    }
}

/// Euclidean (non-periodic) distance metric.
///
/// This is the default distance metric that computes straight-line
/// distances without any boundary wrapping. The implementation is
/// trivial and should compile to identical code as the original
/// non-generic implementation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Euclidean;

impl DistanceMetric for Euclidean {
    const IS_PERIODIC: bool = false;

    #[inline(always)]
    fn delta(&self, a: f32, b: f32, _dim: usize) -> f32 {
        a - b
    }
}

/// Periodic boundary conditions using minimum image convention.
///
/// For molecular dynamics simulations with periodic boxes, particles
/// may be closer through the periodic boundary than in direct space.
/// This metric wraps distances to find the minimum image.
///
/// Currently supports orthorhombic (rectangular) boxes only.
#[derive(Clone, Copy, Debug)]
pub struct Periodic {
    /// Box dimensions [Lx, Ly, Lz] in Angstroms
    pub dimensions: [f32; 3],
    /// Precomputed half dimensions for minimum image check
    half_dimensions: [f32; 3],
    /// Precomputed inverse dimensions for coordinate wrapping
    inv_dimensions: [f32; 3],
}

impl Periodic {
    /// Create a new periodic box with the given dimensions.
    ///
    /// # Arguments
    /// * `dimensions` - Box dimensions [Lx, Ly, Lz] in Angstroms
    ///
    /// # Panics
    /// Panics if any dimension is non-positive.
    pub fn new(dimensions: [f32; 3]) -> Self {
        assert!(
            dimensions[0] > 0.0 && dimensions[1] > 0.0 && dimensions[2] > 0.0,
            "Box dimensions must be positive"
        );
        Self {
            dimensions,
            half_dimensions: [
                dimensions[0] * 0.5,
                dimensions[1] * 0.5,
                dimensions[2] * 0.5,
            ],
            inv_dimensions: [
                1.0 / dimensions[0],
                1.0 / dimensions[1],
                1.0 / dimensions[2],
            ],
        }
    }

    /// Get the box dimensions.
    pub const fn dimensions(&self) -> [f32; 3] {
        self.dimensions
    }

    /// Wrap a coordinate into the primary box [0, L).
    #[inline(always)]
    pub fn wrap_coordinate(&self, coord: f32, dim: usize) -> f32 {
        let l = self.dimensions[dim];
        let wrapped = coord - l * (coord * self.inv_dimensions[dim]).floor();
        // Handle edge case where wrapped == l due to floating point
        if wrapped >= l { 0.0 } else { wrapped }
    }

    /// Wrap a position vector into the primary box.
    #[inline(always)]
    pub fn wrap_position(&self, pos: &[f32; 3]) -> [f32; 3] {
        [
            self.wrap_coordinate(pos[0], 0),
            self.wrap_coordinate(pos[1], 1),
            self.wrap_coordinate(pos[2], 2),
        ]
    }
}

impl DistanceMetric for Periodic {
    const IS_PERIODIC: bool = true;

    #[inline(always)]
    fn delta(&self, a: f32, b: f32, dim: usize) -> f32 {
        let mut d = a - b;
        let l = self.dimensions[dim];
        let half_l = self.half_dimensions[dim];

        // Minimum image convention: wrap to [-L/2, L/2)
        if d > half_l {
            d -= l;
        } else if d < -half_l {
            d += l;
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_euclidean_delta() {
        let metric = Euclidean;
        assert_eq!(metric.delta(5.0, 3.0, 0), 2.0);
        assert_eq!(metric.delta(3.0, 5.0, 0), -2.0);
        assert_eq!(metric.delta(0.0, 0.0, 0), 0.0);
    }

    #[test]
    fn test_euclidean_distance_squared() {
        let metric = Euclidean;
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        assert_eq!(metric.distance_squared(&a, &b), 25.0);
    }

    #[test]
    fn test_periodic_no_wrap() {
        let metric = Periodic::new([10.0, 10.0, 10.0]);
        // Points within half-box distance shouldn't wrap
        assert_relative_eq!(metric.delta(5.0, 3.0, 0), 2.0, max_relative = 1e-6);
        assert_relative_eq!(metric.delta(3.0, 5.0, 0), -2.0, max_relative = 1e-6);
    }

    #[test]
    fn test_periodic_wrap_positive() {
        let metric = Periodic::new([10.0, 10.0, 10.0]);
        // a=9, b=1: direct distance is 8, but wrapped is -2 (a is 2 units "behind" b)
        let d = metric.delta(9.0, 1.0, 0);
        assert_relative_eq!(d, -2.0, max_relative = 1e-6);
    }

    #[test]
    fn test_periodic_wrap_negative() {
        let metric = Periodic::new([10.0, 10.0, 10.0]);
        // a=1, b=9: direct distance is -8, but wrapped is 2
        let d = metric.delta(1.0, 9.0, 0);
        assert_relative_eq!(d, 2.0, max_relative = 1e-6);
    }

    #[test]
    fn test_periodic_exactly_half_box() {
        let metric = Periodic::new([10.0, 10.0, 10.0]);
        // At exactly half-box distance, should be 5.0 (or -5.0)
        let d = metric.delta(0.0, 5.0, 0);
        assert_relative_eq!(d.abs(), 5.0, max_relative = 1e-6);
    }

    #[test]
    fn test_periodic_corner_wrap() {
        let metric = Periodic::new([10.0, 10.0, 10.0]);
        let a = [0.5, 0.5, 0.5];
        let b = [9.5, 9.5, 9.5];
        // Each dimension: |0.5 - 9.5| = 9, wrapped = 1
        // Distance squared = 1 + 1 + 1 = 3
        let dist_sq = metric.distance_squared(&a, &b);
        assert_relative_eq!(dist_sq, 3.0, max_relative = 1e-6);
    }

    #[test]
    fn test_periodic_rectangular_box() {
        let metric = Periodic::new([20.0, 10.0, 5.0]);
        // Test wrapping in smallest dimension (Z)
        let d = metric.delta(0.5, 4.5, 2); // Z dimension, box size 5
        // Direct: 0.5 - 4.5 = -4, which is < -2.5, so wrap: -4 + 5 = 1
        assert_relative_eq!(d, 1.0, max_relative = 1e-6);
    }

    #[test]
    fn test_wrap_coordinate() {
        let metric = Periodic::new([10.0, 10.0, 10.0]);
        assert_relative_eq!(metric.wrap_coordinate(5.0, 0), 5.0, max_relative = 1e-6);
        assert_relative_eq!(metric.wrap_coordinate(15.0, 0), 5.0, max_relative = 1e-6);
        assert_relative_eq!(metric.wrap_coordinate(-3.0, 0), 7.0, max_relative = 1e-6);
        assert_relative_eq!(metric.wrap_coordinate(10.0, 0), 0.0, max_relative = 1e-6);
    }

    #[test]
    #[should_panic(expected = "Box dimensions must be positive")]
    fn test_periodic_invalid_dimensions() {
        Periodic::new([10.0, 0.0, 10.0]);
    }
}
