use pyo3::prelude::*;
use kd_tree::KdPoint;

#[pyclass]
#[derive(Clone)]
pub struct PrimitiveAtom {

    #[pyo3(get, set)]
    pub primitive_type: String,

    #[pyo3(get, set)]
    pub tag: String,

    #[pyo3(get, set)]
    pub coordinates: [f64; 3]
}

#[pymethods]
impl PrimitiveAtom {

    #[new]
    fn new(primitive_type: String, tag: String, coordinates: [f64; 3]) -> Self {
        Self { primitive_type, tag, coordinates }
    }
}

// Needed for the creation of KdTrees.
impl KdPoint for &PrimitiveAtom {
    type Dim = typenum::U3;
    type Scalar = f64;
    fn at(&self, i: usize) -> Self::Scalar { self.coordinates[i] }
}