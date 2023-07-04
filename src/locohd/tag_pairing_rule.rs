use std::collections::HashSet;

use pyo3::prelude::*;

#[derive(FromPyObject, Clone)]
pub enum TagPairingRuleVariants {

    WithoutList { 
        accept_same: bool
    },

    WithList {
        tag_pairs: HashSet<(String, String)>,
        accepted_pairs: bool,
        ordered: bool
    },
}

#[pyclass]
#[derive(Clone)]
pub struct TagPairingRule {
    variant: TagPairingRuleVariants
}

#[pymethods]
impl TagPairingRule {

    #[new]
    pub fn build(variant: TagPairingRuleVariants) -> PyResult<Self> {
        Ok(Self { variant })
    }
}

impl TagPairingRule {

    pub fn pair_accepted(&self, pair: &(String, String)) -> bool {
        
        match &self.variant {

            TagPairingRuleVariants::WithoutList { 
                accept_same 
            } => {
                let mut accepted = pair.0 == pair.1;
                if !accept_same { accepted = !accepted }
                accepted
            },

            TagPairingRuleVariants::WithList { 
                tag_pairs, 
                accepted_pairs, 
                ordered 
            } => {

                let mut accepted = tag_pairs.contains(pair);
                if !ordered { accepted |=  tag_pairs.contains(&(pair.1.clone(), pair.0.clone())) }
                if !accepted_pairs { accepted = !accepted }
                accepted

            },
        }

    }    
}