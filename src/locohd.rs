use std::{ptr, collections::HashMap};

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use rayon::iter::ParallelIterator;

use kd_tree::KdTree;

mod weight_function;
pub use weight_function::WeightFunction;

mod tag_pairing_rule;
pub use tag_pairing_rule::{TagPairingRule, TagPairingRuleVariants};

mod pmf;
use pmf::PMFSystem;

mod primitive_atom;
pub use primitive_atom::PrimitiveAtom;

mod utils;

#[derive(FromPyObject, IntoPyObject, Clone)]
/// Decides whether a single WeightFunction or multiple WeightFunctions are used.
pub enum WeightFunctionOptions {
    Single(WeightFunction),
    Multiple(HashMap<String, WeightFunction>)
}

#[derive(FromPyObject, IntoPyObject, Clone)]
/// Decides whether the anchor pairs use a single, shared weight function or a per-anchor specified one.
pub enum AnchorPairSpecifier {
    WithWeightFunctionKey(Vec<(usize, usize, String)>),
    WithoutWeightFunctionKey(Vec<(usize, usize)>)
}

#[pyclass]
pub struct LoCoHD {

    #[pyo3(get)]
    categories: HashMap<String, usize>,
    #[pyo3(get)]
    category_weights: Vec<f64>,
    #[pyo3(get)]
    w_func: WeightFunctionOptions,
    #[pyo3(get)]
    tag_pairing_rule: TagPairingRule,
    thread_pool: ThreadPool
}

// Methods unavailable from Python.
impl LoCoHD {

    /// Calculates the hellinger integral between two environments belonging to two anchor points.
    pub fn hellinger_integral(&self, 
        seq_a: &Vec<String>, 
        seq_b: &Vec<String>, 
        dists_a: &Vec<f64>, 
        dists_b: &Vec<f64>,
        w_func: &WeightFunction
    ) -> PyResult<f64> {

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
        let mut pmf_system = PMFSystem::new(&self.categories, &self.category_weights);
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
            let delta_w = w_func.integral_range(dist_buffer, new_dist)?;
            h_integral +=  delta_w * current_hdist;
            dist_buffer = new_dist;
        }

        // Finalizing loops. This happens if one of the lists is finished before the other.
        if idx_b < seq_b.len() - 1 {

            let current_hdist = pmf_system.hellinger_dist()?;

            // In this case, the dists_a list is surely finished (see prev. while loop condition), but
            // the dists_b list is not.
            idx_b += 1;
            let delta_w = w_func.integral_range(dists_a[dists_a.len() - 1], dists_b[idx_b])?;
            h_integral += delta_w * current_hdist;

            pmf_system.update_pmf2(&seq_b[idx_b])?;

            // Finishing the dists_b list.
            while idx_b < seq_b.len() - 1 {

                idx_b += 1;

                let current_hdist = pmf_system.hellinger_dist()?;
                let delta_w = w_func.integral_range(dists_b[idx_b - 1], dists_b[idx_b])?;
                h_integral += delta_w * current_hdist;

                pmf_system.update_pmf2(&seq_b[idx_b])?;
            }

            // Last integral until infinity.
            let current_hdist = pmf_system.hellinger_dist()?;
            let delta_w = w_func.integral_range(dists_b[dists_b.len() - 1], f64::INFINITY)?;
            h_integral += delta_w * current_hdist;

        } else if idx_a < seq_a.len() - 1 {

            let current_hdist = pmf_system.hellinger_dist()?;

            // In this case, the dists_b list is finished, but dists_a is not.
            idx_a += 1;
            let delta_w = w_func.integral_range(dists_b[dists_b.len() - 1], dists_a[idx_a])?;
            h_integral += delta_w * current_hdist;

            pmf_system.update_pmf1(&seq_a[idx_a])?;

            // Finishing the dists_a list.
            while idx_a < seq_a.len() - 1 {

                idx_a += 1;

                let current_hdist = pmf_system.hellinger_dist()?;
                let delta_w = w_func.integral_range(dists_a[idx_a - 1], dists_a[idx_a])?;
                h_integral += delta_w * current_hdist;

                pmf_system.update_pmf1(&seq_a[idx_a])?;
            }

            // Last integral until infinity.
            let current_hdist = pmf_system.hellinger_dist()?;
            let delta_w = w_func.integral_range(dists_a[dists_a.len() - 1], f64::INFINITY)?;
            h_integral += delta_w * current_hdist;

        } else if idx_a == seq_a.len() - 1 && idx_b == seq_b.len() - 1 {

            // Last integral until infinity.
            // In this case, dists_a[dists_a.len() - 1] == dists_b[dists_b.len() - 1]
            let current_hdist = pmf_system.hellinger_dist()?;
            let delta_w = w_func.integral_range(dists_a[dists_a.len() - 1], f64::INFINITY)?;
            h_integral += delta_w * current_hdist;

        } else { unreachable!(); }

        Ok(h_integral)
    }

    /// Check for the correct pairing of the WeightFunctionOptions set for self and the w_func_keys.
    /// If correct, it maps these keys (if they exist) to the correct WeightFunctions.
    fn keys_to_weight_functions(&self, 
        w_func_keys: &Option<Vec<String>>,
        target_len: usize
    ) -> PyResult<Vec<&WeightFunction>> {

        match (&self.w_func, w_func_keys) {

            // First correct pairing: multiple WF options with a given key vector.
            (
                WeightFunctionOptions::Multiple(key_to_fn), 
                Some(key_vec)
            ) => {

                if key_vec.len() != target_len {

                    let err_msg = format!(
                        "The w_func_keys vector has an invalid length ({} instead of {})!",
                        key_vec.len(), target_len
                    );
                    return Err(PyValueError::new_err(err_msg));
                }

                let (retrieved, failed): (Vec<_>, Vec<_>) = key_vec.iter()
                    .map(|k| key_to_fn.get(k))
                    .partition(Option::is_some);

                if failed.len() > 0 {

                    let err_msg = format!(
                        "The vector contains {} out of {} invalid weight function keys!",
                        failed.len(), key_vec.len()
                    );
                    return Err(PyValueError::new_err(err_msg));

                } else { 
                    Ok(retrieved.iter().map(|x| x.unwrap()).collect()) 
                }
            },

            // Second correct pairing: a single WF option without a key vector.
            (
                WeightFunctionOptions::Single(w_func),
                None
            ) => Ok(vec![w_func; target_len]),

            // Everything else is invalid!
            _ => {

                let err_msg = format!(
                    "Invalid pairing for the LoCoHD instance's 
                    w_func option and the method's w_func_keys parameter!"
                );
                return Err(PyValueError::new_err(err_msg));
            }
        }
    }

}

#[pymethods]
impl LoCoHD {

    #[new]
    #[pyo3(signature = (categories, w_func=None, tag_pairing_rule=None, n_of_threads=None, category_weights=None))]
    pub fn build(
        categories: Vec<String>, 
        w_func: Option<WeightFunctionOptions>, 
        tag_pairing_rule: Option<TagPairingRule>,
        n_of_threads: Option<usize>,
        category_weights: Option<Vec<f64>>
    ) -> PyResult<Self> {

        // Check categories length validity
        if categories.len() == 0 {
            let err_msg = format!(
                "The number of possible categories (primitive types) cannot be zero!"
            );
            return Err(PyValueError::new_err(err_msg));            
        }

        // For faster lookup of the categories we use a HashMap instead of the supplied Vec.
        let categories: HashMap<_, _> = categories
            .into_iter()
            .enumerate()
            .map(|(a, b)| (b, a))
            .collect();

        // Set default category_weights if necessary
        let category_weights: Vec<f64> = match category_weights {
            None => vec![1.0; categories.len()],
            Some(v) => v
        };

        // Check category_weights length validity
        if category_weights.len() != categories.len() {
            let err_msg = format!(
                "LoCoHD parameters 'categories' and 'category_weights' must have the same lengths! 
                Instead, they have lengths of {} vs. {}!", 
                categories.len(), category_weights.len()
            );
            return Err(PyValueError::new_err(err_msg));
        }

        // Check category_weights positivity validity
        let non_positive_category_weights_len = category_weights.iter()
            .filter(|&&x| x <= 0.0)
            .collect::<Vec<_>>()
            .len();
        if non_positive_category_weights_len > 0 {
            let err_msg = format!(
                "LoCoHD parameter 'category_weights' must only contain positive values! 
                Instead, it contains {} non-positive values!", 
                non_positive_category_weights_len
            );
            return Err(PyValueError::new_err(err_msg));
        }

        // Set default WeightFunctionOptions if necessary
        let w_func = match w_func {
            None => WeightFunctionOptions::Single(WeightFunction::build("uniform".to_owned(), vec![3., 10.])?),
            Some(v) => v
        };

        // Set default TagPairingRule options if necessary
        let tag_pairing_rule = match tag_pairing_rule {
            None => TagPairingRule::build(TagPairingRuleVariants::WithoutList { accept_same: true })?,
            Some(v) => v
        };

        // Set multithreading options
        let n_of_threads = match n_of_threads {
            Some(n) => n,
            None => 0
        };
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(n_of_threads)
            .build()
            .or_else(|err| {
                let err_msg = format!("Error while building the thread pool!\nDebug: {:?}", err);
                return Err(PyValueError::new_err(err_msg));
            })?;

        Ok(Self { categories, category_weights, w_func, tag_pairing_rule, thread_pool })
    }

    /// Wrapper for the hellinger_integral function unavailable from Python.
    #[pyo3(signature = (seq_a, seq_b, dists_a, dists_b, w_func_key = None))]
    pub fn from_anchors(&self, 
        seq_a: Vec<String>, 
        seq_b: Vec<String>, 
        dists_a: Vec<f64>, 
        dists_b: Vec<f64>,
        w_func_key: Option<String>
    ) -> PyResult<f64> {

        let w_func = self.keys_to_weight_functions(
            &w_func_key.map(|x| vec![x; 1]), 
            1
        )?[0];
        self.hellinger_integral(&seq_a, &seq_b, &dists_a, &dists_b, w_func)
    }

    /// Compares two structures with a given sequence pair of categories (seq_a and seq_b) 
    /// and a given distance matrix pair (dmx_a and dmx_b).
    #[pyo3(signature = (seq_a, seq_b, dmx_a, dmx_b, w_func_keys = None))]
    fn from_dmxs(&self, 
        seq_a: Vec<String>, 
        seq_b: Vec<String>, 
        dmx_a: Vec<Vec<f64>>, 
        dmx_b: Vec<Vec<f64>>,
        w_func_keys: Option<Vec<String>>
    ) -> PyResult<Vec<f64>> {

        // Check distance matrix length equivalence.
        if dmx_a.len() != dmx_b.len() {

            let err_msg = format!(
                "Expected matrices with the same length, got lengths {} and {}!",
                dmx_a.len(), 
                dmx_b.len()
            );
            return Err(PyValueError::new_err(err_msg));
        }

        // Create the vector corresponding for the per-anchor WeightFunctions.
        let w_func_vec = self.keys_to_weight_functions(&w_func_keys, dmx_a.len())?;

        // Construct the LoCoHD parallel calculator closure.
        let run_calculation = || dmx_a
            .par_iter()
            .zip(&dmx_b)
            .zip(w_func_vec)
            .map(|((row_a, row_b), w_func)| {
                let (current_dists_a, current_seq_a) = utils::sort_together(row_a, &seq_a);
                let (current_dists_b, current_seq_b) = utils::sort_together(row_b, &seq_b);
                self.hellinger_integral(&current_seq_a, &current_seq_b, &current_dists_a, &current_dists_b, w_func)
            })
            .partition::<Vec<_>, Vec<_>, _>(Result::is_ok);

        // Run calculations in parallel.
        let (output_ok, output_err) = self.thread_pool.install(run_calculation);

        if output_err.len() > 0 {
            let err_msg = "The hellinger_integral 
                function returned an error during the LoCoHD calculations!".to_owned();
            return Err(PyValueError::new_err(err_msg));
        }

        let output_ok = output_ok.into_iter().map(|x| x.unwrap()).collect();

        Ok(output_ok)

    }

    /// Compares two structures with a given sequence pair of categories (coords_a and coords_b) 
    /// and a given coordinate-set pair (dmx_a and dmx_b). It calculates the distance matrices
    /// with the L2 (Euclidean) metric.
    #[pyo3(signature = (seq_a, seq_b, coords_a, coords_b, w_func_keys = None))]
    fn from_coords(&self, 
        seq_a: Vec<String>, 
        seq_b: Vec<String>, 
        coords_a: Vec<[f64; 3]>, 
        coords_b: Vec<[f64; 3]>,
        w_func_keys: Option<Vec<String>>
    ) -> PyResult<Vec<f64>> {

        let dmx_a = utils::calculate_distance_matrix(&coords_a);
        let dmx_b = utils::calculate_distance_matrix(&coords_b);

        self.from_dmxs(seq_a, seq_b, dmx_a, dmx_b, w_func_keys)
    }

    /// Compares two structures with a given primitive atom sequence pair.
    fn from_primitives(&self, 
        prim_a: Vec<PrimitiveAtom>, 
        prim_b: Vec<PrimitiveAtom>, 
        anchor_pairs: AnchorPairSpecifier, 
        threshold_distance: f64
    ) -> PyResult<Vec<f64>> {

        // Create the vector corresponding for the per-anchor WeightFunctions.
        let (anchor_index_pairs, anchor_w_funcs) = match anchor_pairs {
            AnchorPairSpecifier::WithWeightFunctionKey(triples) => {

                let anchor_index_pairs = triples.iter().map(|x| (x.0, x.1)).collect::<Vec<_>>();
                let w_func_keys = triples.into_iter().map(|x| x.2).collect::<Vec<_>>();
                let target_len = w_func_keys.len();
                (anchor_index_pairs, self.keys_to_weight_functions(&Some(w_func_keys), target_len)?)
            },
            AnchorPairSpecifier::WithoutWeightFunctionKey(pairs) => {
                let target_len = pairs.len();
                (pairs, self.keys_to_weight_functions(&None, target_len)?)
            }
        };

        // Build KdTrees from the first and second structures. Since the KdPoint trait is
        // implemented on the &PrimitiveAtom type (note the reference!) it is possible to
        // use prim_a and prim_b (almost) directly.
        let kdtree_a = KdTree::build_by_ordered_float(
            prim_a.iter().collect::<Vec<_>>()
        );

        let kdtree_b = KdTree::build_by_ordered_float(
            prim_b.iter().collect::<Vec<_>>()
        );

        // Define the closure that calculates the co-sorted environmental primitive types and
        // distances measured from an anchor atom.
        let env_from_idx = |
            prim_seq: &Vec<PrimitiveAtom>, 
            anchor_idx: usize, 
            kdtree: &KdTree<&PrimitiveAtom>
        | {

            // Search for the neighbour (environment) atoms of the anchor atom.
            let neighbours = kdtree.within_radius(&prim_seq[anchor_idx].coordinates, threshold_distance);

            // Filter out unwanted contacts based on the tag field and using the tag_pairing_rule field.
            let neighbours = neighbours.into_iter().filter(|&&p| {
                let mut accepted = ptr::eq(p, &prim_seq[anchor_idx]);
                accepted |= self.tag_pairing_rule.pair_accepted(&(prim_seq[anchor_idx].tag.clone(), p.tag.clone()));
                accepted
            });

            // Initialize the primitive type sequence and distances for the environment.
            let mut env_seq = vec![];
            let mut env_dists = vec![];

            // Collect the primitive type sequence and distances for the environment.
            for env_neighbour in neighbours {
                env_seq.push(env_neighbour.primitive_type.clone());
                env_dists.push(utils::euclidean_distance(prim_seq[anchor_idx].coordinates, env_neighbour.coordinates));
            }

            // Sort the distances and sequence together.
            utils::sort_together(&env_dists, &env_seq)
        };

        // This closure will calculate the environment LoCoHD scores paralelly for each anchor pair.
        let run_calculation = || anchor_index_pairs
            .par_iter()
            .zip(anchor_w_funcs)
            .map(|(&(idx1, idx2), w_func)| {

                let (env_a_dists, env_a_seq) = env_from_idx(&prim_a, idx1, &kdtree_a);
                let (env_b_dists, env_b_seq) = env_from_idx(&prim_b, idx2, &kdtree_b);
                self.hellinger_integral(&env_a_seq, &env_b_seq, &env_a_dists, &env_b_dists, w_func)
            })
            .partition::<Vec<_>, Vec<_>, _>(Result::is_ok);

        // Run the closure installed to the thread pool.
        let (output_ok, output_err) = self.thread_pool.install(run_calculation);

        if output_err.len() > 0 {
            let err_msg = format!(
                "The from_anchors function returned an error during the LoCoHD calculations!"
            );
            return Err(PyValueError::new_err(err_msg));
        }

        let output_ok = output_ok.into_iter().map(|x| x.unwrap()).collect();

        Ok(output_ok)
    }
}