mod weight_functions;

use std::collections::HashMap;

use weight_functions::WeightFunction;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
struct PrimitiveAtom {

    #[pyo3(get, set)]
    primitive_type: String,

    #[pyo3(get, set)]
    id: String,

    #[pyo3(get, set)]
    coordinates: Vec<f64>
}

#[pymethods]
impl PrimitiveAtom {

    #[new]
    fn __new__(primitive_type: String, id: String, coordinates: Vec<f64>) -> Self {
        Self { primitive_type, id, coordinates }
    }
}


fn hellinger_dist(counts_a: &Vec<usize>, counts_b: &Vec<usize>) -> f64 {
    let mut norm_a: usize = 0;
    for element in counts_a {
        norm_a += element;
    }

    let mut norm_b: usize = 0;
    for element in counts_b {
        norm_b += element;
    }

    let mut distance: f64 = 0f64;
    for idx in 0..counts_a.len() {
        let p_a = counts_a[idx] as f64 / norm_a as f64;
        let p_b = counts_b[idx] as f64 / norm_b as f64;
        distance += (p_a.sqrt() - p_b.sqrt()).powf(2.0);
    }
    distance = (distance / 2f64).sqrt();

    distance
}

fn euclidean_distance(vec_a: &Vec<f64>, vec_b: &Vec<f64>) -> f64 {

    assert_eq!(vec_a.len(), vec_b.len(), 
        "vec_a and vec_b must have same dimensions, but got instead {} and {}!", vec_a.len(), vec_b.len());
            
    let mut distance = 0.;
    for (&item_a, &item_b) in vec_a.iter().zip(vec_b) {
        distance += (item_a - item_b).powf(2.);
    }
    distance.powf(0.5)
}

// Define the parallel_sort function, which co-sorts a distance matrix line with a list of categories.
fn parallel_sort(dists: &Vec<f64>, cats: &Vec<String>) -> (Vec<f64>, Vec<String>) {

    let mut mask = (0..dists.len()).collect::<Vec<usize>>();
    mask.sort_by(|&idx1, &idx2| dists[idx1].partial_cmp(&dists[idx2]).unwrap());

    let mut new_dmx_line = vec![];
    let mut new_cat = vec![];

    for &idx in mask.iter() {
        new_dmx_line.push(dists[idx]);
        new_cat.push(cats[idx].clone());
    }

    (new_dmx_line, new_cat)
    
}

#[pyclass]
struct LoCoHD {
    #[pyo3(get, set)]
    categories: Vec<String>,
    integrator: WeightFunction
}

#[pymethods]
impl LoCoHD {

    #[new]
    fn __new__(categories: Vec<String>, integrator: (String, Vec<f64>)) -> Self {
        Self { 
            categories: categories, 
            integrator: WeightFunction::new(integrator.0, integrator.1) 
        }
    }

    /// test_integrator(x, /)
    /// --
    /// 
    /// Tests the integrated weight function on a list of values x. Returns the integrated
    /// value from 0 to x.
    fn test_integrator(&self, x: Vec<f64>) -> Vec<f64> {
        let mut output = vec![];
        for item in x {
            assert!(item >= 0., "All tested values must be greater than or equal to 0! Value {} violated this!", item);
            output.push(self.integrator.integral(Some(item)));
        }
        output
    }

    /// from_anchors(seq_a, seq_b, dists_a, dists_b, /)
    /// --
    /// 
    /// Calculates the hellinger integral between two environments belonging to two anchor points.
    fn from_anchors(&self, seq_a: Vec<String>, seq_b: Vec<String>, dists_a: Vec<f64>, dists_b: Vec<f64>) -> f64 {

        // Check input validity.
        assert_eq!(seq_a.len(), dists_a.len(),
            "Lists seq_a and dists_a must have equal lengths!"
        );
        assert_eq!(seq_b.len(), dists_b.len(),
            "Lists seq_b and dists_b must have equal lengths!"
        );
        assert_eq!(dists_a[0], 0f64,
            "The dists_a list must start with a distance of 0!"
        );
        assert_eq!(dists_b[0], 0f64,
            "The dists_a list must start with a distance of 0!"
        );

        // Define the find_cat_idx function, which returns the index of the given
        // category (from now on: CAT) in the categories vector.
        let find_cat_idx = |cat: &String| self.categories.iter().position(|x| *x == *cat).unwrap();

        // Define the delta integrated weight function.
        let delta_w = |x_to: Option<f64>, x_from: Option<f64>| {
            self.integrator.integral(x_to) - self.integrator.integral(x_from)
        };

        // Create the probability mass functions (PMFs) and add the first CAT-observations
        // to them (the first element from seq_a and seq_b).
        let mut counts_a: Vec<usize> = vec![0; self.categories.len()];
        let resi_idx_a: usize = find_cat_idx(&seq_a[0]);
        counts_a[resi_idx_a] += 1;

        let mut counts_b: Vec<usize> = vec![0; self.categories.len()];
        let resi_idx_b: usize = find_cat_idx(&seq_b[0]);
        counts_b[resi_idx_b] += 1;

        // Initialize the parallel indices, the hellinger integral, and a buffer for the previous distance. 
        let mut idx_a: usize = 0;
        let mut idx_b: usize = 0;
        let mut h_integral: f64 = 0.0;
        let mut dist_buffer: f64 = 0.0;

        // Main loop. This collates (like in merge sort) the distances, while calculating the hellinger integral.
        while idx_a < seq_a.len() - 1 && idx_b < seq_b.len() - 1 {

            // Calculate Hellinger distance.            
            let current_hdist = hellinger_dist(&counts_a, &counts_b);

            // Select the next smallest distance from dists_a and dists_b.
            // Also, add the new observed CAT to the PMFs.
            let new_dist = if dists_a[idx_a + 1] < dists_b[idx_b + 1] {

                idx_a += 1;

                let resi_idx_a: usize = find_cat_idx(&seq_a[idx_a]);
                counts_a[resi_idx_a] += 1;

                dists_a[idx_a]

            } else if dists_a[idx_a + 1] > dists_b[idx_b + 1] {

                idx_b += 1;

                let resi_idx_b: usize = find_cat_idx(&seq_b[idx_b]);
                counts_b[resi_idx_b] += 1;

                dists_b[idx_b]

            } else if dists_a[idx_a + 1] == dists_b[idx_b + 1] {

                idx_a += 1;
                idx_b += 1;

                let resi_idx_a: usize = find_cat_idx(&seq_a[idx_a]);
                counts_a[resi_idx_a] += 1;

                let resi_idx_b: usize = find_cat_idx(&seq_b[idx_b]);
                counts_b[resi_idx_b] += 1;

                dists_a[idx_a]
            } else { unimplemented!(); };

            // Increment the integral and assign the distance buffer to the new distance.
            h_integral += delta_w(Some(new_dist), Some(dist_buffer)) * current_hdist;
            dist_buffer = new_dist;
        }

        // Finalizing loops. This happens if one of the lists is finished before the other.
        if idx_b < seq_b.len() - 1 {

            // In this case, the dists_a list is surely finished (see prev. while loop condition), but
            // the dists_b list is not.
            idx_b += 1;
            let current_hdist = hellinger_dist(&counts_a, &counts_b);
            h_integral += delta_w(Some(dists_b[idx_b]), Some(dists_a[dists_a.len() - 1])) * current_hdist;

            let resi_idx_b: usize = find_cat_idx(&seq_b[idx_b]);
            counts_b[resi_idx_b] += 1;

            // Finishing the dists_b list.
            while idx_b < seq_b.len() - 1 {

                idx_b += 1;

                let current_hdist = hellinger_dist(&counts_a, &counts_b);
                h_integral += delta_w(Some(dists_b[idx_b]), Some(dists_b[idx_b - 1])) * current_hdist;

                let resi_idx_b: usize = find_cat_idx(&seq_b[idx_b]);
                counts_b[resi_idx_b] += 1;

            }

            // Last integral until infinity.
            let current_hdist = hellinger_dist(&counts_a, &counts_b);
            h_integral += delta_w(None, Some(dists_b[dists_b.len() - 1])) * current_hdist;

        } else if idx_a < seq_a.len() - 1 {

            // In this case, the dists_b list is finished, but dists_a is not.
            idx_a += 1;
            let current_hdist = hellinger_dist(&counts_a, &counts_b);
            h_integral += delta_w(Some(dists_a[idx_a]), Some(dists_b[dists_b.len() - 1])) * current_hdist;

            let resi_idx_a: usize = find_cat_idx(&seq_a[idx_a]);
            counts_a[resi_idx_a] += 1;

            // Finishing the dists_a list.
            while idx_a < seq_a.len() - 1 {

                idx_a += 1;

                let current_hdist = hellinger_dist(&counts_a, &counts_b);
                h_integral += delta_w(Some(dists_a[idx_a]), Some(dists_a[idx_a - 1])) * current_hdist;

                let resi_idx_a: usize = find_cat_idx(&seq_a[idx_a]);
                counts_a[resi_idx_a] += 1;

            }

            // Last integral until infinity.
            let current_hdist = hellinger_dist(&counts_a, &counts_b);
            h_integral += delta_w(None, Some(dists_a[dists_a.len() - 1])) * current_hdist;

        } else if idx_a == seq_a.len() - 1 && idx_b == seq_b.len() - 1 {

            // Last integral until infinity.
            // In this case, dists_a[dists_a.len() - 1] == dists_b[dists_b.len() - 1]
            let current_hdist = hellinger_dist(&counts_a, &counts_b);
            h_integral += delta_w(None, Some(dists_a[dists_a.len() - 1])) * current_hdist;

        } else { unimplemented!(); }

        h_integral
    }

    /// from_dmxs(seq_a, seq_b, dmx_a, dmx_b, /)
    /// --
    /// 
    /// Compares two structures with a given sequence pair of categories (seq_a and seq_b) 
    /// and a given distance matrix pair (dmx_a and dmx_b). 
    fn from_dmxs(&self, seq_a: Vec<String>, seq_b: Vec<String>, dmx_a: Vec<Vec<f64>>, dmx_b: Vec<Vec<f64>>) -> Vec<f64> {

        // Check input validity.
        assert_eq!(dmx_a.len(), dmx_b.len(),
            "Only structures with the same size are comparable!"
        );

        // Create the line-by-line comparison of the two distance matrices.
        let mut output = vec![];

        for idx in 0..dmx_a.len() {

            let (new_dmx_line_a, new_seq_a) = parallel_sort(&dmx_a[idx], &seq_a);
            let (new_dmx_line_b, new_seq_b) = parallel_sort(&dmx_b[idx], &seq_b);
            output.push(self.from_anchors(new_seq_a, new_seq_b, new_dmx_line_a, new_dmx_line_b));
        }

        output
    }

    /// from_coords(seq_a, seq_b, coords_a, coords_b, /)
    /// --
    /// 
    /// Compares two structures with a given sequence pair of categories (coords_a and coords_b) 
    /// and a given coordinate-set pair (dmx_a and dmx_b). It calculates the distance matrices
    /// with the p2 (Euclidean) metric.
    fn from_coords(&self, seq_a: Vec<String>, seq_b: Vec<String>, coords_a: Vec<Vec<f64>>, coords_b: Vec<Vec<f64>>) -> Vec<f64> {

        let calculate_dmx = |coords: &Vec<Vec<f64>>| {

            let mut distance_mx = vec![vec![0.; coords.len()]; coords.len()];
            
            for idx1 in 0..coords.len() {
                for idx2 in idx1 + 1..coords.len() {
                    let distance = euclidean_distance(&coords[idx1], &coords[idx2]);
                    distance_mx[idx1][idx2] = distance;
                    distance_mx[idx2][idx1] = distance;
                }
            }
            distance_mx
        };

        self.from_dmxs(seq_a, seq_b, calculate_dmx(&coords_a), calculate_dmx(&coords_b))

    }

    fn from_primitives(&self, 
        prim_a: Vec<PrimitiveAtom>, 
        prim_b: Vec<PrimitiveAtom>, 
        anchor_pairs: Vec<(usize, usize)>, 
        only_hetero_contacts: bool,
        threshold_distance: f64) -> Vec<f64> {

        let mut dmx_a: HashMap<(usize, usize), f64> = HashMap::new();
        let mut dmx_b: HashMap<(usize, usize), f64> = HashMap::new();

        let mut output = vec![];
        
        for &(idx_a1, idx_b1) in anchor_pairs.iter() {

            let mut dists_a: Vec<f64> = vec![];
            let mut seq_a: Vec<String> = vec![];

            for idx_a2 in 0..prim_a.len() {

                if idx_a1 == idx_a2 {
                    dists_a.push(0.);
                    seq_a.push(prim_a[idx_a1].primitive_type.clone());
                    continue;
                }

                if prim_a[idx_a1].id == prim_a[idx_a2].id && only_hetero_contacts {
                    continue;
                }

                let idx_pair = if idx_a1 > idx_a2 { (idx_a2, idx_a1) } else { (idx_a1, idx_a2) };

                let dist = match dmx_a.get(&idx_pair) {
                    Some(&dist) => dist,
                    None => {
                        let dist = euclidean_distance(&prim_a[idx_a1].coordinates, &prim_a[idx_a2].coordinates);
                        dmx_a.insert(idx_pair, dist);
                        dist
                    }
                };

                if dist > threshold_distance {
                    continue;
                }

                dists_a.push(dist);                
                seq_a.push(prim_a[idx_a2].primitive_type.clone());
            }

            let mut dists_b: Vec<f64> = vec![];
            let mut seq_b: Vec<String> = vec![];

            for idx_b2 in 0..prim_b.len() {

                if idx_b1 == idx_b2 {
                    dists_b.push(0.);
                    seq_b.push(prim_b[idx_b1].primitive_type.clone());
                    continue;
                }

                if prim_b[idx_b1].id == prim_b[idx_b2].id && only_hetero_contacts {
                    continue;
                }

                let idx_pair = if idx_b1 > idx_b2 { (idx_b2, idx_b1) } else { (idx_b1, idx_b2) };

                let dist = match dmx_b.get(&idx_pair) {
                    Some(&dist) => dist,
                    None => {
                        let dist = euclidean_distance(&prim_b[idx_b1].coordinates, &prim_b[idx_b2].coordinates);
                        dmx_b.insert(idx_pair, dist);
                        dist
                    }
                };

                if dist > threshold_distance {
                    continue;
                }

                dists_b.push(dist);
                seq_b.push(prim_b[idx_b2].primitive_type.clone());
            }

            let (dists_a, seq_a) = parallel_sort(&dists_a, &seq_a);
            let (dists_b, seq_b) = parallel_sort(&dists_b, &seq_b);

            output.push(
                self.from_anchors(seq_a, seq_b, dists_a, dists_b)
            );
        }

        output
    }
}

#[pymodule]
fn loco_hd(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LoCoHD>()?;
    m.add_class::<PrimitiveAtom>()?;
    Ok(())
}
