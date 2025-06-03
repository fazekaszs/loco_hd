use std::collections::HashMap;
use pyo3::{prelude::*, exceptions::PyValueError};

mod statistical_distances;
pub use statistical_distances::StatisticalDistance;

/// The Probability Mass Function System is used to capture the
/// LoCoHD primitive types ("categories" to avoid confusion with
/// the programming primitive types) and to implement the logic
/// behind PMF manipulations: addition to a PMF through the update
/// functions, and calculation of the Hellinger-distance between the
/// two stored PMFs.
pub struct PMFSystem<'a> {
    categories: &'a HashMap<String, usize>,
    category_weights: &'a Vec<f64>,
    pmf1: Vec<f64>,
    pmf2: Vec<f64>
}

impl<'a> PMFSystem<'a> {

    pub fn new(
        categories: &'a HashMap<String, usize>, 
        category_weights: &'a Vec<f64>
    ) -> PMFSystem<'a> {
        PMFSystem {
            categories,
            category_weights,
            pmf1: vec![0f64; categories.len()],
            pmf2: vec![0f64; categories.len()]
        }
    }

    fn find_category_idx(&self, category: &String) -> PyResult<usize> {

        let category_idx = self.categories.get(category);

        let category_idx = if let Some(&idx) = category_idx { idx }
        else { 
            let err_msg = format!("Category (with name {}) not found!", category);
            return Err(PyValueError::new_err(err_msg)); 
        };

        Ok(category_idx)
    }

    pub fn update_pmf1(&mut self, category: &String) -> PyResult<()> {

        let category_idx = self.find_category_idx(category)?;
            
        self.pmf1[category_idx] += self.category_weights[category_idx];

        Ok(())
    }

    pub fn update_pmf2(&mut self, category: &String) -> PyResult<()> {

        let category_idx = self.find_category_idx(category)?;
            
        self.pmf2[category_idx] += self.category_weights[category_idx];

        Ok(())
    }

    pub fn get_normalized_form(&self) -> PyResult<(Vec<f64>, Vec<f64>)> {

        let norm1 = self.pmf1.iter().sum::<f64>();
        let norm2 = self.pmf2.iter().sum::<f64>();

        if norm1 == 0f64 { 
            let err_msg = "Zero norm error for PMF1".to_owned();
            return Err(PyValueError::new_err(err_msg)); 
        } else if norm2 == 0f64 { 
            let err_msg = "Zero norm error for PMF2".to_owned();
            return Err(PyValueError::new_err(err_msg)); 
        }

        Ok((
            self.pmf1.clone().into_iter().map(|x| x / norm1).collect(),
            self.pmf2.clone().into_iter().map(|x| x / norm2).collect()
        ))

    }

    pub fn calculate_distance(&self, statistical_distance: &StatisticalDistance) -> PyResult<f64> {
        let (p1, p2) = self.get_normalized_form()?;
        statistical_distance.run(p1, p2)
    }
}