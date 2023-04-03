use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct PrimitiveAtom {

    #[pyo3(get, set)]
    pub primitive_type: String,

    #[pyo3(get, set)]
    pub tag: String,

    #[pyo3(get, set)]
    pub coordinates: Vec<f64>
}

#[pymethods]
impl PrimitiveAtom {

    #[new]
    fn new(primitive_type: String, tag: String, coordinates: Vec<f64>) -> Self {
        Self { primitive_type, tag, coordinates }
    }
}