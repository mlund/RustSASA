// Copyright (c) 2024 Maxwell Campbell. Licensed under the MIT License.
//! Tests for periodic boundary condition support in SASA calculations.

use approx::assert_relative_eq;
use rust_sasa::structures::periodic::{DistanceMetric, Euclidean, Periodic};
use rust_sasa::{Atom, calculate_sasa_internal, calculate_sasa_with_pbc};
use std::f32::consts::PI;

const PROBE_RADIUS: f32 = 1.4;
const HIGH_PRECISION_N_POINTS: usize = 50000;
const RELATIVE_TOLERANCE: f32 = 0.005;

fn create_atom(x: f32, y: f32, z: f32, radius: f32, id: usize) -> Atom {
    Atom {
        position: [x, y, z],
        radius,
        id,
        parent_id: None,
    }
}

// ============ Regression Tests: Ensure non-PBC path is unchanged ============

#[test]
fn test_regression_single_sphere() {
    let atoms = vec![create_atom(0.0, 0.0, 0.0, 2.0, 1)];

    // Old API (must remain unchanged)
    let sasa_old = calculate_sasa_internal(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1);

    // New API with Euclidean metric should produce identical results
    let sasa_new = calculate_sasa_with_pbc(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1, None);

    // Must be bit-identical
    assert_eq!(
        sasa_old[0].to_bits(),
        sasa_new[0].to_bits(),
        "Non-PBC path must produce bit-identical results"
    );
}

#[test]
fn test_regression_overlapping_spheres() {
    let atoms = vec![
        create_atom(0.0, 0.0, 0.0, 2.0, 1),
        create_atom(3.0, 0.0, 0.0, 2.0, 2),
    ];

    let sasa_old = calculate_sasa_internal(&atoms, PROBE_RADIUS, 5000, 1);
    let sasa_new = calculate_sasa_with_pbc(&atoms, PROBE_RADIUS, 5000, 1, None);

    // Bit-identical check
    assert_eq!(sasa_old[0].to_bits(), sasa_new[0].to_bits());
    assert_eq!(sasa_old[1].to_bits(), sasa_new[1].to_bits());
}

#[test]
fn test_regression_three_spheres() {
    let atoms = vec![
        create_atom(0.0, 0.0, 0.0, 2.0, 1),
        create_atom(5.0, 0.0, 0.0, 2.0, 2),
        create_atom(10.0, 0.0, 0.0, 2.0, 3),
    ];

    let sasa_old = calculate_sasa_internal(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1);
    let sasa_new = calculate_sasa_with_pbc(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1, None);

    for i in 0..3 {
        assert_eq!(
            sasa_old[i].to_bits(),
            sasa_new[i].to_bits(),
            "Atom {} must have identical SASA",
            i
        );
    }
}

// ============ PBC-specific Tests ============

#[test]
fn test_single_sphere_periodic_same_as_nonperiodic() {
    let radius = 2.0;
    let atoms = vec![create_atom(5.0, 5.0, 5.0, radius, 1)];
    let pbox = Periodic::new([20.0, 20.0, 20.0]);

    let sasa_periodic =
        calculate_sasa_with_pbc(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1, Some(pbox));
    let sasa_nonperiodic =
        calculate_sasa_internal(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1);

    // Single isolated atom should have same SASA regardless of PBC
    assert_relative_eq!(
        sasa_periodic[0],
        sasa_nonperiodic[0],
        max_relative = RELATIVE_TOLERANCE
    );
}

#[test]
fn test_two_spheres_across_periodic_boundary() {
    let radius = 2.0;
    let pbox = Periodic::new([10.0, 10.0, 10.0]);

    // Atoms at x=1 and x=9 are 8 apart directly, but 2 apart through boundary
    let atoms = vec![
        create_atom(1.0, 5.0, 5.0, radius, 1),
        create_atom(9.0, 5.0, 5.0, radius, 2),
    ];

    let sasa_periodic =
        calculate_sasa_with_pbc(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1, Some(pbox));

    // With effective radius 3.4 (2.0 + 1.4), atoms at distance 2 should overlap
    let r = radius + PROBE_RADIUS;
    let full_area = 4.0 * PI * r * r;

    // Both atoms should have reduced SASA due to overlap
    assert!(
        sasa_periodic[0] < full_area * 0.95,
        "Expected overlap to reduce SASA, got {} vs full {}",
        sasa_periodic[0],
        full_area
    );
    assert!(
        sasa_periodic[1] < full_area * 0.95,
        "Expected overlap to reduce SASA"
    );
}

#[test]
fn test_two_spheres_no_overlap_without_pbc_but_overlap_with() {
    let radius = 2.0;
    let pbox = Periodic::new([10.0, 10.0, 10.0]);

    // Atoms at x=0.5 and x=9.5: direct distance 9, PBC distance 1
    let atoms = vec![
        create_atom(0.5, 5.0, 5.0, radius, 1),
        create_atom(9.5, 5.0, 5.0, radius, 2),
    ];

    // Without PBC: atoms are 9 apart, effective radius sum = 6.8, no overlap
    let sasa_no_pbc =
        calculate_sasa_with_pbc(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1, None);

    // With PBC: atoms are 1 apart through boundary, significant overlap
    let sasa_pbc =
        calculate_sasa_with_pbc(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1, Some(pbox));

    let r = radius + PROBE_RADIUS;
    let full_area = 4.0 * PI * r * r;

    // Without PBC: full SASA (no overlap)
    assert_relative_eq!(sasa_no_pbc[0], full_area, max_relative = RELATIVE_TOLERANCE);

    // With PBC: reduced SASA (overlap through boundary)
    assert!(
        sasa_pbc[0] < full_area * 0.8,
        "With PBC, atoms should significantly overlap through boundary"
    );
}

#[test]
fn test_pbc_overlap_matches_direct_overlap() {
    let radius = 2.0;
    let pbox = Periodic::new([10.0, 10.0, 10.0]);

    // Atoms 1 apart through periodic boundary
    let atoms_pbc = vec![
        create_atom(0.5, 5.0, 5.0, radius, 1),
        create_atom(9.5, 5.0, 5.0, radius, 2),
    ];

    // Create equivalent non-periodic setup with atoms at distance 1
    let atoms_direct = vec![
        create_atom(0.0, 0.0, 0.0, radius, 1),
        create_atom(1.0, 0.0, 0.0, radius, 2),
    ];

    let sasa_pbc = calculate_sasa_with_pbc(
        &atoms_pbc,
        PROBE_RADIUS,
        HIGH_PRECISION_N_POINTS,
        1,
        Some(pbox),
    );
    let sasa_direct =
        calculate_sasa_internal(&atoms_direct, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1);

    // SASA should be similar (same overlap geometry)
    assert_relative_eq!(
        sasa_pbc[0],
        sasa_direct[0],
        max_relative = RELATIVE_TOLERANCE
    );
    assert_relative_eq!(
        sasa_pbc[1],
        sasa_direct[1],
        max_relative = RELATIVE_TOLERANCE
    );
}

#[test]
fn test_rectangular_box_z_wrap() {
    let radius = 1.5;
    let pbox = Periodic::new([20.0, 20.0, 5.0]); // Small Z dimension

    // Atoms close through the Z boundary
    let atoms = vec![
        create_atom(10.0, 10.0, 0.5, radius, 1),
        create_atom(10.0, 10.0, 4.5, radius, 2), // 4 apart direct, 1 through boundary
    ];

    let sasa =
        calculate_sasa_with_pbc(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1, Some(pbox));

    let r = radius + PROBE_RADIUS;
    let full_area = 4.0 * PI * r * r;

    // Should overlap since they're only 1 Angstrom apart through boundary
    assert!(sasa[0] < full_area * 0.9, "Expected overlap in Z dimension");
}

#[test]
fn test_self_interaction_prevention() {
    // Ensure an atom doesn't "see itself" through the periodic boundary
    let radius = 2.0;
    let pbox = Periodic::new([5.0, 5.0, 5.0]); // Small box

    let atoms = vec![create_atom(2.5, 2.5, 2.5, radius, 1)];

    let sasa =
        calculate_sasa_with_pbc(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1, Some(pbox));

    // Single atom should have full SASA (no self-occlusion)
    let r = radius + PROBE_RADIUS;
    let expected_area = 4.0 * PI * r * r;
    assert_relative_eq!(sasa[0], expected_area, max_relative = RELATIVE_TOLERANCE);
}

#[test]
fn test_corner_atom_multiple_images() {
    let radius = 1.5;
    let pbox = Periodic::new([10.0, 10.0, 10.0]);

    // Atom near corner at (0.5, 0.5, 0.5)
    // Another atom near opposite corner at (9.5, 9.5, 9.5)
    // Through PBC they're sqrt(3) ≈ 1.73 apart
    let atoms = vec![
        create_atom(0.5, 0.5, 0.5, radius, 1),
        create_atom(9.5, 9.5, 9.5, radius, 2),
    ];

    let sasa =
        calculate_sasa_with_pbc(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1, Some(pbox));

    let r = radius + PROBE_RADIUS;
    let full_area = 4.0 * PI * r * r;

    // Effective radius sum = 5.8, PBC distance = sqrt(3) ≈ 1.73
    // They should overlap
    assert!(
        sasa[0] < full_area * 0.95,
        "Corner atoms should overlap through PBC"
    );
}

#[test]
fn test_large_box_no_pbc_effect() {
    let radius = 2.0;
    let pbox = Periodic::new([100.0, 100.0, 100.0]); // Very large box

    let atoms = vec![
        create_atom(10.0, 10.0, 10.0, radius, 1),
        create_atom(20.0, 10.0, 10.0, radius, 2),
    ];

    let sasa_pbc =
        calculate_sasa_with_pbc(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1, Some(pbox));
    let sasa_no_pbc =
        calculate_sasa_with_pbc(&atoms, PROBE_RADIUS, HIGH_PRECISION_N_POINTS, 1, None);

    // With large box, PBC should have no effect
    assert_relative_eq!(
        sasa_pbc[0],
        sasa_no_pbc[0],
        max_relative = RELATIVE_TOLERANCE
    );
    assert_relative_eq!(
        sasa_pbc[1],
        sasa_no_pbc[1],
        max_relative = RELATIVE_TOLERANCE
    );
}

// ============ Tests for Distance Metric directly ============

#[test]
fn test_euclidean_vs_periodic_in_center() {
    let euclidean = Euclidean;
    let periodic = Periodic::new([10.0, 10.0, 10.0]);

    let a = [3.0, 3.0, 3.0];
    let b = [5.0, 5.0, 5.0];

    // Both metrics should give same result for points well within box
    let dist_e = euclidean.distance_squared(&a, &b);
    let dist_p = periodic.distance_squared(&a, &b);

    assert_relative_eq!(dist_e, dist_p, max_relative = 1e-6);
}

#[test]
fn test_periodic_distance_across_boundary() {
    let periodic = Periodic::new([10.0, 10.0, 10.0]);

    let a = [1.0, 5.0, 5.0];
    let b = [9.0, 5.0, 5.0];

    // Direct distance squared = 64, PBC distance squared = 4
    let dist_sq = periodic.distance_squared(&a, &b);
    assert_relative_eq!(dist_sq, 4.0, max_relative = 1e-6);
}
