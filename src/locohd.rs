use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use rayon::prelude::*;
use rayon::iter::ParallelIterator;

use kd_tree::KdTree;

#[cfg(test)]
mod weight_function_utests;
mod weight_function;
pub use weight_function::WeightFunction;

#[cfg(test)]
mod pmf_utests;
mod pmf;
use pmf::PMFSystem;

#[cfg(test)]
mod locohd_utests;

mod primitive_atom;
pub use primitive_atom::PrimitiveAtom;

mod utils;

#[pyclass]
pub struct LoCoHD {

    #[pyo3(get, set)]
    categories: Vec<String>,

    w_func: WeightFunction
}

#[pymethods]
impl LoCoHD {

    #[new]
    pub fn new(categories: Vec<String>, w_func: WeightFunction) -> Self {
        Self { categories, w_func }
    }

    /// Calculates the hellinger integral between two environments belonging to two anchor points.
    pub fn from_anchors(&self, 
        seq_a: Vec<String>, 
        seq_b: Vec<String>, 
        dists_a: Vec<f64>, 
        dists_b: Vec<f64>) -> PyResult<f64> 
    {

        // Check input validity.
        if seq_a.len() != dists_a.len() || seq_b.len() != dists_b.len() {
            let err_msg = "Lists seq and dists must have equal lengths!".to_owned();
            return Err(PyValueError::new_err(err_msg));
        }
        if dists_a[0] != 0. || dists_b[0] != 0. {
            let err_msg = "The dists list must start with a distance of 0!".to_owned();
            return Err(PyValueError::new_err(err_msg));
        }

        // Create the probability mass functions (PMFs) and add the first CAT-observations
        // to them (the first element from seq_a and seq_b).
        let mut pmf_system = PMFSystem::new(&self.categories);
        pmf_system.update_pmf1(&seq_a[0])?;
        pmf_system.update_pmf2(&seq_b[0])?;

        // Initialize the parallel indices, the hellinger integral, and a buffer for the previous distance. 
        let mut idx_a: usize = 0;
        let mut idx_b: usize = 0;
        let mut h_integral: f64 = 0.;
        let mut dist_buffer: f64 = 0.;

        // Main loop. This collates (like in merge sort) the distances, while calculating the hellinger integral.
        while idx_a < seq_a.len() - 1 && idx_b < seq_b.len() - 1 {

            // Calculate Hellinger distance.            
            let current_hdist = pmf_system.hellinger_dist()?;

            // Select the next smallest distance from dists_a and dists_b.
            // Also, add the new observed CAT to the PMFs.
            let new_dist = if dists_a[idx_a + 1] < dists_b[idx_b + 1] {

                idx_a += 1;
                pmf_system.update_pmf1(&seq_a[idx_a])?;
                dists_a[idx_a]

            } else if dists_a[idx_a + 1] > dists_b[idx_b + 1] {

                idx_b += 1;
                pmf_system.update_pmf2(&seq_b[idx_b])?;
                dists_b[idx_b]

            } else if dists_a[idx_a + 1] == dists_b[idx_b + 1] {

                idx_a += 1;
                idx_b += 1;
                pmf_system.update_pmf1(&seq_a[idx_a])?;
                pmf_system.update_pmf2(&seq_b[idx_b])?;
                dists_a[idx_a]

            } else { unreachable!(); };

            // Increment the integral and assign the distance buffer to the new distance.
            let delta_w = self.w_func.integral_range(dist_buffer, new_dist)?;
            h_integral +=  delta_w * current_hdist;
            dist_buffer = new_dist;
        }

        // Finalizing loops. This happens if one of the lists is finished before the other.
        if idx_b < seq_b.len() - 1 {

            let current_hdist = pmf_system.hellinger_dist()?;

            // In this case, the dists_a list is surely finished (see prev. while loop condition), but
            // the dists_b list is not.
            idx_b += 1;
            let delta_w = self.w_func.integral_range(dists_a[dists_a.len() - 1], dists_b[idx_b])?;
            h_integral += delta_w * current_hdist;

            pmf_system.update_pmf2(&seq_b[idx_b])?;

            // Finishing the dists_b list.
            while idx_b < seq_b.len() - 1 {

                idx_b += 1;

                let current_hdist = pmf_system.hellinger_dist()?;
                let delta_w = self.w_func.integral_range(dists_b[idx_b - 1], dists_b[idx_b])?;
                h_integral += delta_w * current_hdist;

                pmf_system.update_pmf2(&seq_b[idx_b])?;
            }

            // Last integral until infinity.
            let current_hdist = pmf_system.hellinger_dist()?;
            let delta_w = self.w_func.integral_range(dists_b[dists_b.len() - 1], f64::INFINITY)?;
            h_integral += delta_w * current_hdist;

        } else if idx_a < seq_a.len() - 1 {

            let current_hdist = pmf_system.hellinger_dist()?;

            // In this case, the dists_b list is finished, but dists_a is not.
            idx_a += 1;
            let delta_w = self.w_func.integral_range(dists_b[dists_b.len() - 1], dists_a[idx_a])?;
            h_integral += delta_w * current_hdist;

            pmf_system.update_pmf1(&seq_a[idx_a])?;

            // Finishing the dists_a list.
            while idx_a < seq_a.len() - 1 {

                idx_a += 1;

                let current_hdist = pmf_system.hellinger_dist()?;
                let delta_w = self.w_func.integral_range(dists_a[idx_a - 1], dists_a[idx_a])?;
                h_integral += delta_w * current_hdist;

                pmf_system.update_pmf1(&seq_a[idx_a])?;
            }

            // Last integral until infinity.
            let current_hdist = pmf_system.hellinger_dist()?;
            let delta_w = self.w_func.integral_range(dists_a[dists_a.len() - 1], f64::INFINITY)?;
            h_integral += delta_w * current_hdist;

        } else if idx_a == seq_a.len() - 1 && idx_b == seq_b.len() - 1 {

            // Last integral until infinity.
            // In this case, dists_a[dists_a.len() - 1] == dists_b[dists_b.len() - 1]
            let current_hdist = pmf_system.hellinger_dist()?;
            let delta_w = self.w_func.integral_range(dists_a[dists_a.len() - 1], f64::INFINITY)?;
            h_integral += delta_w * current_hdist;

        } else { unimplemented!(); }

        Ok(h_integral)
    }

    /// Compares two structures with a given sequence pair of categories (seq_a and seq_b) 
    /// and a given distance matrix pair (dmx_a and dmx_b).
    fn from_dmxs(&self, 
        seq_a: Vec<String>, 
        seq_b: Vec<String>, 
        dmx_a: Vec<Vec<f64>>, 
        dmx_b: Vec<Vec<f64>>) -> PyResult<Vec<f64>>
    {

        // Check input validity.
        if dmx_a.len() != dmx_b.len() {

            let err_msg = format!(
                "Expected matrices with the same lengt, got lengths {} and {}!", 
                dmx_a.len(), 
                dmx_b.len()
            );
            return Err(PyValueError::new_err(err_msg));
        }

        let (output_ok, output_err): (Vec<_>, Vec<_>) = dmx_a
            .par_iter()
            .zip(&dmx_b)
            .map(|(row_a, row_b)| {
                let (current_dists_a, current_seq_a) = utils::sort_together(row_a, &seq_a);
                let (current_dists_b, current_seq_b) = utils::sort_together(row_b, &seq_b);
                self.from_anchors(current_seq_a, current_seq_b, current_dists_a, current_dists_b)
            })
            .partition(Result::is_ok);

        if output_err.len() > 0 {
            let err_msg = "The from_anchors function returned an error during the LoCoHD calculations!".to_owned();
            return Err(PyValueError::new_err(err_msg));
        }

        let output_ok = output_ok.into_iter().map(|x| x.unwrap()).collect();

        Ok(output_ok)

    }

    /// Compares two structures with a given sequence pair of categories (coords_a and coords_b) 
    /// and a given coordinate-set pair (dmx_a and dmx_b). It calculates the distance matrices
    /// with the L2 (Euclidean) metric.
    fn from_coords(&self, 
        seq_a: Vec<String>, 
        seq_b: Vec<String>, 
        coords_a: Vec<[f64; 3]>, 
        coords_b: Vec<[f64; 3]>) -> PyResult<Vec<f64>>
    {

        let dmx_a = utils::calculate_distance_matrix(&coords_a);
        let dmx_b = utils::calculate_distance_matrix(&coords_b);

        self.from_dmxs(seq_a, seq_b, dmx_a, dmx_b)
    }

    /// Compares two structures with a given primitive atom sequence pair.
    fn from_primitives(&self, 
        prim_a: Vec<PrimitiveAtom>, 
        prim_b: Vec<PrimitiveAtom>, 
        anchor_pairs: Vec<(usize, usize)>, 
        only_hetero_contacts: bool,
        threshold_distance: f64) -> PyResult<Vec<f64>>
    {

        // Build KdTrees from the first and second structures. Since the KdPoint trait is
        // implemented on the &PrimitiveAtom type (note the reference!) it is possible to
        // use prim_a and prim_b (almost) directly.
        let kdtree_a = KdTree::build_by_ordered_float(
            prim_a.iter().collect::<Vec<_>>()
        );

        let kdtree_b = KdTree::build_by_ordered_float(
            prim_b.iter().collect::<Vec<_>>()
        );

        // Calculate the environment LoCoHD scores paralelly for each anchor pair.
        let (output_ok, output_err): (Vec<_>, Vec<_>) = anchor_pairs
            .par_iter()
            .map(|&(idx1, idx2)| {

                // Search for the neighbour (environment) atoms of the anchor atom in the first structure.
                // Also, filter out homo residue contacts if necessary (based on the tag field).
                let neighbours_a = kdtree_a.within_radius(&prim_a[idx1].coordinates, threshold_distance);
                let neighbours_a = neighbours_a.into_iter().filter(|&p| 
                    p.tag != prim_a[idx1].tag || !only_hetero_contacts
                );

                // Initialize the primitive type sequence and distances for the first environment.
                let mut env_a_seq = vec![];
                let mut env_a_dists = vec![];

                // If only_hetero_contacts is true, then we have to put back the anchor atom
                // into the environment, since we have just filtered it out.
                if only_hetero_contacts {
                    env_a_seq.push(prim_a[idx1].primitive_type.clone());
                    env_a_dists.push(0f64);
                }

                // Collect the primitive type sequence and distances for the first environment.
                for env_a_neighbour in neighbours_a {
                    env_a_seq.push(env_a_neighbour.primitive_type.clone());
                    env_a_dists.push(utils::euclidean_distance(prim_a[idx1].coordinates, env_a_neighbour.coordinates));
                }

                // Sort the distances and sequence together.
                let (env_a_dists, env_a_seq) = utils::sort_together(&env_a_dists, &env_a_seq);

                // Do the same thing with the second environment.

                let neighbours_b = kdtree_b.within_radius(&prim_b[idx2].coordinates, threshold_distance);
                let neighbours_b = neighbours_b.into_iter().filter(|&p| 
                    p.tag != prim_b[idx2].tag || !only_hetero_contacts
                );

                let mut env_b_seq = vec![];
                let mut env_b_dists = vec![];

                if only_hetero_contacts {
                    env_b_seq.push(prim_b[idx2].primitive_type.clone());
                    env_b_dists.push(0f64);
                }

                for env_b_neighbour in neighbours_b {
                    env_b_seq.push(env_b_neighbour.primitive_type.clone());
                    env_b_dists.push(utils::euclidean_distance(prim_b[idx2].coordinates, env_b_neighbour.coordinates));
                }

                let (env_b_dists, env_b_seq) = utils::sort_together(&env_b_dists, &env_b_seq);
                
                // Return the LoCoHD score.
                self.from_anchors(env_a_seq, env_b_seq, env_a_dists, env_b_dists)
        })
        .partition(Result::is_ok);

        if output_err.len() > 0 {
            let err_msg = "The from_anchors function returned an error during the LoCoHD calculations!".to_owned();
            return Err(PyValueError::new_err(err_msg));
        }

        let output_ok = output_ok.into_iter().map(|x| x.unwrap()).collect();

        Ok(output_ok)
    }
}