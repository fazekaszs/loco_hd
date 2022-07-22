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

fn w_integral(x: Option<f64>, w_params_a: &Vec<f64>, w_params_b: &Vec<f64>) -> f64 {

    // Calculates 1 - SUM_i(A_i * exp(-B_i * x)) / SUM_i(A_i)
    // which is the CDF of the hyperexponential distribution.

    if x == None {
        return 1.0;
    }
    let x = x.unwrap();

    let mut norm = 0.;
    let mut sum = 0.;

    for (&item_a, &item_b) in w_params_a.iter().zip(w_params_b.iter()) {
        sum += item_a * (- item_b * x).exp();
        norm += item_a;
    }

    1. - sum / norm
}

#[pyclass]
struct LoCoHD {
    categories: Vec<String>,
    w_params_a: Vec<f64>,
    w_params_b: Vec<f64>
}

#[pymethods]
impl LoCoHD {

    #[new]
    fn new(categories: Vec<String>, w_params_a: Vec<f64>, w_params_b: Vec<f64>) -> Self {

        assert_eq!(w_params_a.len(), w_params_b.len(), 
            "An equal number of weight parameters must be given!"
        );

        Self { categories, w_params_a, w_params_b }
    }

    #[getter]
    fn get_categories(&self) -> Vec<String> {
        self.categories.clone()
    }

    #[setter]
    fn set_categories(&mut self, categories: Vec<String>) {
        self.categories = categories
    }

    fn hellinger_integral(self_: PyRef<'_, Self>, resi_a: Vec<String>, resi_b: Vec<String>, dists_a: Vec<f64>, dists_b: Vec<f64>) -> f64 {

        // Check input validity.
        assert_eq!(resi_a.len(), dists_a.len(),
            "Lists resi_a and dists_a must have equal lengths!"
        );
        assert_eq!(resi_b.len(), dists_b.len(),
            "Lists resi_b and dists_b must have equal lengths!"
        );
        assert_eq!(dists_a[0], 0f64,
            "The dists_a list must start with a distance of 0!"
        );
        assert_eq!(dists_b[0], 0f64,
            "The dists_a list must start with a distance of 0!"
        );

        // Define the find_cat_idx function, which returns the index of the given
        // category (from now on: CAT) in the categories vector.
        let find_cat_idx = |cat: &String| self_.categories.iter().position(|x| *x == *cat).unwrap();

        // Define the delta weight function.
        let delta_w = |x_to: Option<f64>, x_from: Option<f64>| {
            w_integral(x_to, &self_.w_params_a, &self_.w_params_b) - w_integral(x_from, &self_.w_params_a, &self_.w_params_b)
        };

        // Create the probability mass functions (PMFs) and add the first CAT-observations
        // to them (the first element from resi_a and resi_b).
        let mut counts_a: Vec<usize> = vec![0; self_.categories.len()];
        let resi_idx_a: usize = find_cat_idx(&resi_a[0]);
        counts_a[resi_idx_a] += 1;

        let mut counts_b: Vec<usize> = vec![0; self_.categories.len()];
        let resi_idx_b: usize = find_cat_idx(&resi_b[0]);
        counts_b[resi_idx_b] += 1;

        // Initialize the parallel indices, the hellinger integral, and a buffer for the previous distance. 
        let mut idx_a: usize = 0;
        let mut idx_b: usize = 0;
        let mut h_integral: f64 = 0.0;
        let mut dist_buffer: f64 = 0.0;

        // Main loop. This collates (like in merge sort) the distances, while calculating the hellinger integral.
        while idx_a < resi_a.len() - 1 && idx_b < resi_b.len() - 1 {

            // Calculate Hellinger distance.            
            let current_hdist = hellinger_dist(&counts_a, &counts_b);

            // Select the next smallest distance from dists_a and dists_b.
            // Also, add the new observed CAT to the PMFs.
            let new_dist = if dists_a[idx_a + 1] < dists_b[idx_b + 1] {

                idx_a += 1;

                let resi_idx_a: usize = find_cat_idx(&resi_a[idx_a]);
                counts_a[resi_idx_a] += 1;

                dists_a[idx_a]

            } else if dists_a[idx_a + 1] > dists_b[idx_b + 1] {

                idx_b += 1;

                let resi_idx_b: usize = find_cat_idx(&resi_b[idx_b]);
                counts_b[resi_idx_b] += 1;

                dists_b[idx_b]

            } else if dists_a[idx_a + 1] == dists_b[idx_b + 1] {

                idx_a += 1;
                idx_b += 1;

                let resi_idx_a: usize = find_cat_idx(&resi_a[idx_a]);
                counts_a[resi_idx_a] += 1;

                let resi_idx_b: usize = find_cat_idx(&resi_b[idx_b]);
                counts_b[resi_idx_b] += 1;

                dists_a[idx_a]
            } else { unimplemented!(); };

            // Increment the integral and assign the distance buffer to the new distance.
            h_integral += delta_w(Some(new_dist), Some(dist_buffer)) * current_hdist;
            dist_buffer = new_dist;
        }

        // Finalizing loops. This happens if one of the lists is finished before the other.
        if idx_b < resi_b.len() - 1 {

            // In this case, the dists_a list is surely finished (see prev. while loop condition), but
            // the dists_b list is not.
            idx_b += 1;
            let current_hdist = hellinger_dist(&counts_a, &counts_b);
            h_integral += delta_w(Some(dists_b[idx_b]), Some(dists_a[dists_a.len() - 1])) * current_hdist;

            let resi_idx_b: usize = find_cat_idx(&resi_b[idx_b]);
            counts_b[resi_idx_b] += 1;

            // Finishing the dists_b list.
            while idx_b < resi_b.len() - 1 {

                idx_b += 1;

                let current_hdist = hellinger_dist(&counts_a, &counts_b);
                h_integral += delta_w(Some(dists_b[idx_b]), Some(dists_b[idx_b - 1])) * current_hdist;

                let resi_idx_b: usize = find_cat_idx(&resi_b[idx_b]);
                counts_b[resi_idx_b] += 1;

            }

            // Last integral until infinity.
            let current_hdist = hellinger_dist(&counts_a, &counts_b);
            h_integral += delta_w(None, Some(dists_b[dists_b.len() - 1])) * current_hdist;

        } else if idx_a < resi_a.len() - 1 {

            // In this case, the dists_b list is finished, but dists_a is not.
            idx_a += 1;
            let current_hdist = hellinger_dist(&counts_a, &counts_b);
            h_integral += delta_w(Some(dists_a[idx_a]), Some(dists_b[dists_b.len() - 1])) * current_hdist;

            let resi_idx_a: usize = find_cat_idx(&resi_a[idx_a]);
            counts_a[resi_idx_a] += 1;

            // Finishing the dists_a list.
            while idx_a < resi_a.len() - 1 {

                idx_a += 1;

                let current_hdist = hellinger_dist(&counts_a, &counts_b);
                h_integral += delta_w(Some(dists_a[idx_a]), Some(dists_a[idx_a - 1])) * current_hdist;

                let resi_idx_a: usize = find_cat_idx(&resi_a[idx_a]);
                counts_a[resi_idx_a] += 1;

            }

            // Last integral until infinity.
            let current_hdist = hellinger_dist(&counts_a, &counts_b);
            h_integral += delta_w(None, Some(dists_a[dists_a.len() - 1])) * current_hdist;

        } else if idx_a == resi_a.len() - 1 && idx_b == resi_b.len() - 1 {

            // Last integral until infinity.
            // In this case, dists_a[dists_a.len() - 1] == dists_b[dists_b.len() - 1]
            let current_hdist = hellinger_dist(&counts_a, &counts_b);
            h_integral += delta_w(None, Some(dists_a[dists_a.len() - 1])) * current_hdist;

        } else { unimplemented!(); }

        h_integral
    }

}

#[pymodule]
fn loco_hd(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LoCoHD>()?;
    Ok(())
}
