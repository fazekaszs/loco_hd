mod weight_functions;

use weight_functions::WeightFunction;
use pyo3::prelude::*;

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

    /// hellinger_integral(seq_a, seq_b, dists_a, dists_b, /)
    /// --
    /// 
    /// Calculates the hellinger integral between two environments belonging to two anchor points.
    fn hellinger_integral(&self, seq_a: Vec<String>, seq_b: Vec<String>, dists_a: Vec<f64>, dists_b: Vec<f64>) -> f64 {

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

    /// compare_structures(seq_a, seq_b, dmx_a, dmx_b, /)
    /// --
    /// 
    /// Compares two structures with a given sequence pair of categories (seq_a and seq_b) 
    /// and a given distance matrix pair (dmx_a and dmx_b). 
    fn compare_structures(&self, seq_a: Vec<String>, seq_b: Vec<String>, dmx_a: Vec<Vec<f64>>, dmx_b: Vec<Vec<f64>>) -> Vec<f64> {

        // Check input validity.
        assert_eq!(dmx_a.len(), dmx_b.len(),
            "Only structures with the same size are comparable!"
        );

        // Define the parallel_sort function, which co-sorts a distance matrix line with a list of categories.
        let parallel_sort = |dmx_line: &Vec<f64>, cat: &Vec<String>| {

            let mut mask = (0..dmx_line.len()).collect::<Vec<usize>>();
            mask.sort_by(|&idx1, &idx2| dmx_line[idx1].partial_cmp(&dmx_line[idx2]).unwrap());

            let mut new_dmx_line = vec![];
            let mut new_cat = vec![];

            for &idx in mask.iter() {
                new_dmx_line.push(dmx_line[idx]);
                new_cat.push(cat[idx].clone());
            }

            (new_dmx_line, new_cat)
        };

        // Create the line-by-line comparison of the two distance matrices.
        let mut output = vec![];

        for idx in 0..dmx_a.len() {

            let (new_dmx_line_a, new_seq_a) = parallel_sort(&dmx_a[idx], &seq_a);
            let (new_dmx_line_b, new_seq_b) = parallel_sort(&dmx_b[idx], &seq_b);
            output.push(self.hellinger_integral(new_seq_a, new_seq_b, new_dmx_line_a, new_dmx_line_b));
        }

        output
    }

}

#[pymodule]
fn loco_hd(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LoCoHD>()?;
    Ok(())
}
