// Copyright (c) 2024 Maxwell Campbell. Licensed under the MIT License.
use crate::structures::periodic::{DistanceMetric, Euclidean, Periodic};
use crate::{Atom, NeighborData};

/// Spatial grid for efficient neighbor finding.
///
/// Generic over `DistanceMetric` to support both Euclidean and periodic
/// boundary conditions. The compiler monomorphizes separate code paths,
/// ensuring zero overhead for the Euclidean case.
pub struct SpatialGrid<D: DistanceMetric = Euclidean> {
    /// Atom indices sorted by cell (contiguous per cell)
    atom_indices: Vec<u32>,

    /// Positions in SoA layout
    positions_x: Vec<f32>,
    positions_y: Vec<f32>,
    positions_z: Vec<f32>,

    /// Radii for each sorted atom (parallel to positions)
    radii: Vec<f32>,

    /// Start index in atom_indices for each cell
    cell_starts: Vec<u32>,

    /// Grid parameters
    grid_dims: [u32; 3],
    num_cells: usize,

    /// Precomputed half-shell offsets for the search extent
    half_shell_offsets: Vec<(i32, i32, i32)>,

    /// Whether to use periodic wrapping for cell lookup
    use_periodic_cells: bool,

    /// The distance metric
    metric: D,
}

impl SpatialGrid<Euclidean> {
    /// Create a new spatial grid with Euclidean (non-periodic) distance metric.
    pub fn new(
        atoms: &[Atom],
        active_indices: &[usize],
        cell_size: f32,
        max_search_radius: f32,
    ) -> Self {
        Self::with_metric(atoms, active_indices, cell_size, max_search_radius, Euclidean)
    }
}

impl SpatialGrid<Periodic> {
    /// Create a new spatial grid with periodic boundary conditions.
    ///
    /// For PBC, the search extent is increased to ensure neighbors across
    /// the periodic boundary are found correctly.
    pub fn new_periodic(
        atoms: &[Atom],
        active_indices: &[usize],
        cell_size: f32,
        max_search_radius: f32,
        periodic: Periodic,
    ) -> Self {
        Self::with_metric_pbc(atoms, active_indices, cell_size, max_search_radius, periodic)
    }

    /// Create a periodic grid with correct search extent for PBC.
    fn with_metric_pbc(
        atoms: &[Atom],
        active_indices: &[usize],
        cell_size: f32,
        max_search_radius: f32,
        metric: Periodic,
    ) -> Self {
        // For PBC, use the periodic box to define grid bounds
        let box_dims = metric.dimensions();

        // Ensure grid covers the entire periodic box
        // Use min_bounds at origin for simplicity
        let min_bounds = [0.0f32; 3];
        let inv_cell_size = 1.0 / cell_size;

        // Grid dimensions based on periodic box
        let grid_dims = [
            ((box_dims[0]) * inv_cell_size).ceil() as u32,
            ((box_dims[1]) * inv_cell_size).ceil() as u32,
            ((box_dims[2]) * inv_cell_size).ceil() as u32,
        ];
        let num_cells = (grid_dims[0] * grid_dims[1] * grid_dims[2]) as usize;

        // For PBC, search extent must cover the maximum distance in each dimension
        // Since max image distance is L/2, and we need to search all neighbors
        // within max_search_radius, the extent needs to be large enough.
        // To be safe, use half the grid dimensions (covers entire half-space).
        let search_extent = {
            let based_on_radius = (max_search_radius / cell_size).ceil() as i32;
            let max_half_dim = (grid_dims[0].max(grid_dims[1]).max(grid_dims[2]) / 2) as i32;
            based_on_radius.max(max_half_dim)
        };

        // Precompute half-shell offsets for this search extent
        let half_shell_offsets = Self::compute_half_shell_offsets(search_extent);

        // Count atoms per cell - wrap positions into the primary box
        let mut cell_counts = vec![0u32; num_cells];
        for &idx in active_indices {
            let wrapped_pos = metric.wrap_position(&atoms[idx].position);
            let cell = Self::get_cell_index_static(
                &wrapped_pos,
                &min_bounds,
                inv_cell_size,
                &grid_dims,
            );
            cell_counts[cell] += 1;
        }

        // Build cell_starts (exclusive prefix sum)
        let mut cell_starts = vec![0u32; num_cells + 1];
        for i in 0..num_cells {
            cell_starts[i + 1] = cell_starts[i] + cell_counts[i];
        }

        // Allocate and fill sorted arrays
        let n_active = active_indices.len();
        let mut atom_indices = vec![0u32; n_active];
        let mut positions_x = vec![0.0f32; n_active];
        let mut positions_y = vec![0.0f32; n_active];
        let mut positions_z = vec![0.0f32; n_active];
        let mut radii = vec![0.0f32; n_active];

        let mut write_pos = cell_starts[..num_cells].to_vec();

        for &orig_idx in active_indices {
            let atom = &atoms[orig_idx];
            let wrapped_pos = metric.wrap_position(&atom.position);
            let cell = Self::get_cell_index_static(&wrapped_pos, &min_bounds, inv_cell_size, &grid_dims);

            let wp = write_pos[cell] as usize;
            atom_indices[wp] = orig_idx as u32;
            // Store original positions (not wrapped) for correct distance calculations
            positions_x[wp] = atom.position[0];
            positions_y[wp] = atom.position[1];
            positions_z[wp] = atom.position[2];
            radii[wp] = atom.radius;

            write_pos[cell] += 1;
        }

        Self {
            atom_indices,
            positions_x,
            positions_y,
            positions_z,
            radii,
            cell_starts,
            grid_dims,
            num_cells,
            half_shell_offsets,
            use_periodic_cells: true,
            metric,
        }
    }
}

impl<D: DistanceMetric> SpatialGrid<D> {
    /// Create a new spatial grid with the given distance metric.
    pub fn with_metric(
        atoms: &[Atom],
        active_indices: &[usize],
        cell_size: f32,
        max_search_radius: f32,
        metric: D,
    ) -> Self {
        // Calculate bounds
        let (min_bounds, max_bounds) = Self::calculate_bounds(atoms, active_indices, cell_size);
        let inv_cell_size = 1.0 / cell_size;

        // Grid dimensions
        let grid_dims = [
            ((max_bounds[0] - min_bounds[0]) * inv_cell_size).ceil() as u32 + 1,
            ((max_bounds[1] - min_bounds[1]) * inv_cell_size).ceil() as u32 + 1,
            ((max_bounds[2] - min_bounds[2]) * inv_cell_size).ceil() as u32 + 1,
        ];
        let num_cells = (grid_dims[0] * grid_dims[1] * grid_dims[2]) as usize;

        // Calculate search extent
        let search_extent = (max_search_radius / cell_size).ceil() as i32;

        // Precompute half-shell offsets for this search extent
        let half_shell_offsets = Self::compute_half_shell_offsets(search_extent);

        // Count atoms per cell
        let mut cell_counts = vec![0u32; num_cells];
        for &idx in active_indices {
            let cell = Self::get_cell_index_static(
                &atoms[idx].position,
                &min_bounds,
                inv_cell_size,
                &grid_dims,
            );
            cell_counts[cell] += 1;
        }

        // Build cell_starts (exclusive prefix sum)
        let mut cell_starts = vec![0u32; num_cells + 1];
        for i in 0..num_cells {
            cell_starts[i + 1] = cell_starts[i] + cell_counts[i];
        }

        // Allocate and fill sorted arrays
        let n_active = active_indices.len();
        let mut atom_indices = vec![0u32; n_active];
        let mut positions_x = vec![0.0f32; n_active];
        let mut positions_y = vec![0.0f32; n_active];
        let mut positions_z = vec![0.0f32; n_active];
        let mut radii = vec![0.0f32; n_active];

        let mut write_pos = cell_starts[..num_cells].to_vec();

        for &orig_idx in active_indices {
            let atom = &atoms[orig_idx];
            let pos = &atom.position;
            let cell = Self::get_cell_index_static(pos, &min_bounds, inv_cell_size, &grid_dims);

            let wp = write_pos[cell] as usize;
            atom_indices[wp] = orig_idx as u32;
            positions_x[wp] = pos[0];
            positions_y[wp] = pos[1];
            positions_z[wp] = pos[2];
            radii[wp] = atom.radius;

            write_pos[cell] += 1;
        }

        // Determine if we should use periodic cell wrapping (compile-time constant)
        let use_periodic_cells = D::IS_PERIODIC;

        Self {
            atom_indices,
            positions_x,
            positions_y,
            positions_z,
            radii,
            cell_starts,
            grid_dims,
            num_cells,
            half_shell_offsets,
            use_periodic_cells,
            metric,
        }
    }

    fn calculate_bounds(
        atoms: &[Atom],
        active_indices: &[usize],
        padding: f32,
    ) -> ([f32; 3], [f32; 3]) {
        let mut min_b = [f32::INFINITY; 3];
        let mut max_b = [f32::NEG_INFINITY; 3];

        for &idx in active_indices {
            let pos = &atoms[idx].position;
            for i in 0..3 {
                min_b[i] = min_b[i].min(pos[i]);
                max_b[i] = max_b[i].max(pos[i]);
            }
        }

        for i in 0..3 {
            min_b[i] -= padding;
            max_b[i] += padding;
        }

        (min_b, max_b)
    }

    #[inline(always)]
    fn get_cell_index_static(
        pos: &[f32; 3],
        min_bounds: &[f32; 3],
        inv_cell_size: f32,
        grid_dims: &[u32; 3],
    ) -> usize {
        let x = ((pos[0] - min_bounds[0]) * inv_cell_size) as u32;
        let y = ((pos[1] - min_bounds[1]) * inv_cell_size) as u32;
        let z = ((pos[2] - min_bounds[2]) * inv_cell_size) as u32;
        (x + y * grid_dims[0] + z * grid_dims[0] * grid_dims[1]) as usize
    }

    #[inline(always)]
    const fn cell_coords_to_index(&self, cx: i32, cy: i32, cz: i32) -> Option<usize> {
        if self.use_periodic_cells {
            // Wrap cell coordinates for periodic boundaries
            let cx = cx.rem_euclid(self.grid_dims[0] as i32) as u32;
            let cy = cy.rem_euclid(self.grid_dims[1] as i32) as u32;
            let cz = cz.rem_euclid(self.grid_dims[2] as i32) as u32;
            Some((cx + cy * self.grid_dims[0] + cz * self.grid_dims[0] * self.grid_dims[1]) as usize)
        } else {
            // Non-periodic: reject out-of-bounds cells
            if cx < 0 || cy < 0 || cz < 0 {
                return None;
            }
            let cx = cx as u32;
            let cy = cy as u32;
            let cz = cz as u32;
            if cx >= self.grid_dims[0] || cy >= self.grid_dims[1] || cz >= self.grid_dims[2] {
                return None;
            }
            Some((cx + cy * self.grid_dims[0] + cz * self.grid_dims[0] * self.grid_dims[1]) as usize)
        }
    }

    #[inline(always)]
    const fn index_to_cell_coords(&self, idx: usize) -> (i32, i32, i32) {
        let idx = idx as u32;
        let cz = idx / (self.grid_dims[0] * self.grid_dims[1]);
        let remainder = idx % (self.grid_dims[0] * self.grid_dims[1]);
        let cy = remainder / self.grid_dims[0];
        let cx = remainder % self.grid_dims[0];
        (cx as i32, cy as i32, cz as i32)
    }

    /// Compute half-shell offsets for a given search extent [See http://doi.acm.org/10.1145/1862648.1862653]
    ///
    /// Half-shell means: for each pair of cells, only one cell "owns" the check.
    /// We select cells where (dz > 0) OR (dz == 0 && dy > 0) OR (dz == 0 && dy == 0 && dx >= 0)
    /// Note: dx >= 0 includes self (0,0,0) which we handle specially
    fn compute_half_shell_offsets(extent: i32) -> Vec<(i32, i32, i32)> {
        let mut offsets = Vec::new();

        for dz in -extent..=extent {
            for dy in -extent..=extent {
                for dx in -extent..=extent {
                    // Half-shell condition
                    let include =
                        (dz > 0) || (dz == 0 && dy > 0) || (dz == 0 && dy == 0 && dx >= 0);

                    if include {
                        offsets.push((dx, dy, dz));
                    }
                }
            }
        }

        offsets
    }

    /// Build neighbor lists for all active atoms
    pub fn build_all_neighbor_lists(
        &self,
        atoms: &[Atom],
        active_indices: &[usize],
        probe_radius: f32,
        max_radius: f32,
    ) -> Vec<Vec<NeighborData>> {
        let n_atoms = atoms.len();
        let n_active = active_indices.len();

        // Map original index -> active index
        let mut orig_to_active = vec![u32::MAX; n_atoms];
        for (active_idx, &orig_idx) in active_indices.iter().enumerate() {
            orig_to_active[orig_idx] = active_idx as u32;
        }

        // Preallocate neighbor lists
        let mut neighbors: Vec<Vec<NeighborData>> =
            (0..n_active).map(|_| Vec::with_capacity(80)).collect();

        // Maximum possible search radius (for quick rejection)
        let max_search_radius = max_radius + max_radius + 2.0 * probe_radius;
        let max_search_radius_sq = max_search_radius * max_search_radius;

        // Iterate through all cells using half-shell pattern
        for cell_a in 0..self.num_cells {
            let start_a = self.cell_starts[cell_a] as usize;
            let end_a = self.cell_starts[cell_a + 1] as usize;

            if start_a == end_a {
                continue;
            }

            let (cx, cy, cz) = self.index_to_cell_coords(cell_a);

            // Process this cell against all cells in half-shell
            for &(dx, dy, dz) in &self.half_shell_offsets {
                let cell_b = match self.cell_coords_to_index(cx + dx, cy + dy, cz + dz) {
                    Some(c) => c,
                    None => continue,
                };

                let start_b = self.cell_starts[cell_b] as usize;
                let end_b = self.cell_starts[cell_b + 1] as usize;

                if start_b == end_b {
                    continue;
                }

                let is_self = dx == 0 && dy == 0 && dz == 0;

                if is_self {
                    self.process_self_cell(
                        atoms,
                        &orig_to_active,
                        start_a,
                        end_a,
                        probe_radius,
                        max_radius,
                        max_search_radius_sq,
                        &mut neighbors,
                    );
                } else {
                    self.process_neighbor_cells(
                        atoms,
                        &orig_to_active,
                        start_a,
                        end_a,
                        start_b,
                        end_b,
                        probe_radius,
                        max_radius,
                        max_search_radius_sq,
                        &mut neighbors,
                    );
                }
            }
        }

        // Sort neighbors by distance for early-exit optimization
        self.sort_neighbors_by_distance(atoms, active_indices, &mut neighbors);

        neighbors
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn process_self_cell(
        &self,
        atoms: &[Atom],
        orig_to_active: &[u32],
        start: usize,
        end: usize,
        probe_radius: f32,
        max_radius: f32,
        max_search_radius_sq: f32,
        neighbors: &mut [Vec<NeighborData>],
    ) {
        for i in start..end {
            let orig_i = self.atom_indices[i] as usize;
            let active_i = orig_to_active[orig_i];
            if active_i == u32::MAX {
                continue;
            }

            let pos_i = [
                self.positions_x[i],
                self.positions_y[i],
                self.positions_z[i],
            ];
            let ri = self.radii[i];
            let id_i = atoms[orig_i].id;

            // Search radius for atom i
            let sr_i = ri + max_radius + 2.0 * probe_radius;
            let sr_i_sq = sr_i * sr_i;

            for j in (i + 1)..end {
                let orig_j = self.atom_indices[j] as usize;

                // Skip if same atom ID
                if atoms[orig_j].id == id_i {
                    continue;
                }

                let pos_j = [
                    self.positions_x[j],
                    self.positions_y[j],
                    self.positions_z[j],
                ];

                // Use metric for distance calculation
                let dist_sq = self.metric.distance_squared(&pos_i, &pos_j);

                // Quick rejection
                if dist_sq > max_search_radius_sq {
                    continue;
                }

                let rj = self.radii[j];

                // Search radius for atom j
                let sr_j = rj + max_radius + 2.0 * probe_radius;
                let sr_j_sq = sr_j * sr_j;

                // Check if i finds j
                if dist_sq <= sr_i_sq {
                    let thresh_j = rj + probe_radius;
                    neighbors[active_i as usize].push(NeighborData {
                        idx: orig_j as u32,
                        threshold_squared: thresh_j * thresh_j,
                    });
                }

                // Check if j finds i
                if dist_sq <= sr_j_sq {
                    let active_j = orig_to_active[orig_j];
                    if active_j != u32::MAX {
                        let thresh_i = ri + probe_radius;
                        neighbors[active_j as usize].push(NeighborData {
                            idx: orig_i as u32,
                            threshold_squared: thresh_i * thresh_i,
                        });
                    }
                }
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn process_neighbor_cells(
        &self,
        atoms: &[Atom],
        orig_to_active: &[u32],
        start_a: usize,
        end_a: usize,
        start_b: usize,
        end_b: usize,
        probe_radius: f32,
        max_radius: f32,
        max_search_radius_sq: f32,
        neighbors: &mut [Vec<NeighborData>],
    ) {
        for i in start_a..end_a {
            let orig_i = self.atom_indices[i] as usize;
            let active_i = orig_to_active[orig_i];
            if active_i == u32::MAX {
                continue;
            }

            let pos_i = [
                self.positions_x[i],
                self.positions_y[i],
                self.positions_z[i],
            ];
            let ri = self.radii[i];
            let id_i = atoms[orig_i].id;

            // Search radius for atom i
            let sr_i = ri + max_radius + 2.0 * probe_radius;
            let sr_i_sq = sr_i * sr_i;

            for j in start_b..end_b {
                let orig_j = self.atom_indices[j] as usize;

                // Skip if same atom ID
                if atoms[orig_j].id == id_i {
                    continue;
                }

                let pos_j = [
                    self.positions_x[j],
                    self.positions_y[j],
                    self.positions_z[j],
                ];

                // Use metric for distance calculation
                let dist_sq = self.metric.distance_squared(&pos_i, &pos_j);

                // Quick rejection
                if dist_sq > max_search_radius_sq {
                    continue;
                }

                let rj = self.radii[j];

                // Search radius for atom j
                let sr_j = rj + max_radius + 2.0 * probe_radius;
                let sr_j_sq = sr_j * sr_j;

                // Check if i finds j
                if dist_sq <= sr_i_sq {
                    let thresh_j = rj + probe_radius;
                    neighbors[active_i as usize].push(NeighborData {
                        idx: orig_j as u32,
                        threshold_squared: thresh_j * thresh_j,
                    });
                }

                // Check if j finds i
                if dist_sq <= sr_j_sq {
                    let active_j = orig_to_active[orig_j];
                    if active_j != u32::MAX {
                        let thresh_i = ri + probe_radius;
                        neighbors[active_j as usize].push(NeighborData {
                            idx: orig_i as u32,
                            threshold_squared: thresh_i * thresh_i,
                        });
                    }
                }
            }
        }
    }

    fn sort_neighbors_by_distance(
        &self,
        atoms: &[Atom],
        active_indices: &[usize],
        neighbors: &mut [Vec<NeighborData>],
    ) {
        for (active_idx, neighbor_list) in neighbors.iter_mut().enumerate() {
            if neighbor_list.len() <= 1 {
                continue;
            }

            let center = atoms[active_indices[active_idx]].position;

            neighbor_list.sort_unstable_by(|a, b| {
                let pa = atoms[a.idx as usize].position;
                let pb = atoms[b.idx as usize].position;

                // Use metric for distance calculation
                let da = self.metric.distance_squared(&center, &pa);
                let db = self.metric.distance_squared(&center, &pb);

                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    /// Get a reference to the distance metric
    pub const fn metric(&self) -> &D {
        &self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Atom;

    fn create_atom(x: f32, y: f32, z: f32, radius: f32, id: usize) -> Atom {
        Atom {
            position: [x, y, z],
            radius,
            id,
            parent_id: None,
        }
    }

    #[test]
    fn test_euclidean_neighbor_finding() {
        let atoms = vec![
            create_atom(0.0, 0.0, 0.0, 1.5, 1),
            create_atom(3.0, 0.0, 0.0, 1.5, 2),
        ];
        let active_indices: Vec<usize> = (0..atoms.len()).collect();
        let probe_radius = 1.4;
        let max_radius = 1.5;
        let cell_size = probe_radius + max_radius;
        let max_search_radius = max_radius + max_radius + 2.0 * probe_radius;

        let grid = SpatialGrid::new(&atoms, &active_indices, cell_size, max_search_radius);
        let neighbors = grid.build_all_neighbor_lists(&atoms, &active_indices, probe_radius, max_radius);

        // Atom 0 should have atom 1 as neighbor
        assert!(!neighbors[0].is_empty(), "Atom 0 should have neighbors");
        assert_eq!(neighbors[0][0].idx, 1, "Atom 0 should have atom 1 as neighbor");

        // Atom 1 should have atom 0 as neighbor
        assert!(!neighbors[1].is_empty(), "Atom 1 should have neighbors");
        assert_eq!(neighbors[1][0].idx, 0, "Atom 1 should have atom 0 as neighbor");
    }

    #[test]
    fn test_periodic_neighbor_finding_direct() {
        // Test atoms that are close in direct space (no wrapping needed)
        let atoms = vec![
            create_atom(3.0, 5.0, 5.0, 1.5, 1),
            create_atom(5.0, 5.0, 5.0, 1.5, 2),
        ];
        let active_indices: Vec<usize> = (0..atoms.len()).collect();
        let probe_radius = 1.4;
        let max_radius = 1.5;
        let cell_size = probe_radius + max_radius;
        let max_search_radius = max_radius + max_radius + 2.0 * probe_radius;
        let pbox = Periodic::new([10.0, 10.0, 10.0]);

        let grid = SpatialGrid::new_periodic(&atoms, &active_indices, cell_size, max_search_radius, pbox);
        let neighbors = grid.build_all_neighbor_lists(&atoms, &active_indices, probe_radius, max_radius);

        // Both atoms should find each other
        assert!(!neighbors[0].is_empty(), "Atom 0 should have neighbors");
        assert!(!neighbors[1].is_empty(), "Atom 1 should have neighbors");
    }

    #[test]
    fn test_periodic_neighbor_finding_across_boundary() {
        // Test atoms that are close through the periodic boundary
        // Atom at x=1 and x=9 with box size 10: distance through boundary is 2
        let atoms = vec![
            create_atom(1.0, 5.0, 5.0, 2.0, 1),
            create_atom(9.0, 5.0, 5.0, 2.0, 2),
        ];
        let active_indices: Vec<usize> = (0..atoms.len()).collect();
        let probe_radius = 1.4;
        let max_radius = 2.0;
        let cell_size = probe_radius + max_radius;
        let max_search_radius = max_radius + max_radius + 2.0 * probe_radius;
        let pbox = Periodic::new([10.0, 10.0, 10.0]);

        // Verify the metric computes correct distance
        let dist_sq = pbox.distance_squared(&atoms[0].position, &atoms[1].position);
        assert!((dist_sq - 4.0).abs() < 0.001, "PBC distance squared should be 4, got {}", dist_sq);

        let grid = SpatialGrid::new_periodic(&atoms, &active_indices, cell_size, max_search_radius, pbox);

        // Check grid properties
        assert!(grid.use_periodic_cells, "Grid should use periodic cells");

        let neighbors = grid.build_all_neighbor_lists(&atoms, &active_indices, probe_radius, max_radius);

        // Verify search radius covers the PBC distance
        let sr = max_radius + max_radius + 2.0 * probe_radius;
        assert!(dist_sq <= sr * sr, "Distance {} should be within search radius {}", dist_sq.sqrt(), sr);

        // Both atoms should find each other through periodic boundary
        assert!(!neighbors[0].is_empty(),
            "Atom 0 should have neighbors (PBC distance is {}, search radius is {})",
            dist_sq.sqrt(), sr);
        assert!(!neighbors[1].is_empty(),
            "Atom 1 should have neighbors");
    }

    #[test]
    fn test_periodic_metric_is_used_in_grid() {
        // Verify the grid is actually using the Periodic metric
        let pbox = Periodic::new([10.0, 10.0, 10.0]);
        let a = [1.0, 5.0, 5.0];
        let b = [9.0, 5.0, 5.0];

        // Euclidean distance squared
        let euclidean = Euclidean;
        let dist_euclidean = euclidean.distance_squared(&a, &b);
        assert!((dist_euclidean - 64.0).abs() < 0.001, "Euclidean should be 64");

        // Periodic distance squared
        let dist_periodic = pbox.distance_squared(&a, &b);
        assert!((dist_periodic - 4.0).abs() < 0.001, "Periodic should be 4");
    }
}
