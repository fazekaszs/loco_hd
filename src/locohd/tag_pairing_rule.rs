use std::collections::HashSet;

use pyo3::prelude::*;

#[derive(FromPyObject, Clone, Debug)]
pub enum TagPairingRuleVariants {

    WithoutList { 
        #[pyo3(item)]
        accept_same: bool
    },

    WithList {
        #[pyo3(item)]
        tag_pairs: HashSet<(String, String)>,
        #[pyo3(item)]
        accepted_pairs: bool,
        #[pyo3(item)]
        ordered: bool
    },
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct TagPairingRule {
    variant: TagPairingRuleVariants
}

#[pymethods]
impl TagPairingRule {

    #[new]
    pub fn build(variant: TagPairingRuleVariants) -> PyResult<Self> {
        Ok(Self { variant })
    }

    #[pyo3(name="pair_accepted")]
    pub fn pair_accepted_py(&self, pair: (String, String)) -> bool {
        self.pair_accepted(&pair)
    }

    pub fn get_dbg_str(&self) -> String {
        format!("{:#?}", self)
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